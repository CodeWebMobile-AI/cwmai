"""
Redis Streams Integration with Existing Worker Intelligence

Seamless integration of Redis Streams with existing CWMAI worker intelligence
components, providing backward compatibility and enhanced capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable, Union
import uuid

from redis_intelligence_hub import RedisIntelligenceHub, IntelligenceEvent, EventType, EventPriority
from redis_event_sourcing import RedisEventStore
from redis_distributed_workflows import RedisWorkflowEngine
from redis_lockfree_adapter import create_lockfree_state_manager

# Import existing intelligence components
try:
    from swarm_intelligence import SwarmIntelligence
    from intelligence_hub import IntelligenceHub
    from ai_brain import AIBrain
    from task_manager import TaskManager
    from god_mode_controller import GodModeController
    LEGACY_COMPONENTS_AVAILABLE = True
except ImportError:
    LEGACY_COMPONENTS_AVAILABLE = False
    logging.warning("Some legacy intelligence components not available for integration")


class RedisIntelligenceIntegrator:
    """Integrates Redis Streams with existing CWMAI intelligence components."""
    
    def __init__(self,
                 enable_legacy_bridge: bool = True,
                 enable_event_migration: bool = True,
                 enable_real_time_sync: bool = True):
        """Initialize Redis intelligence integrator.
        
        Args:
            enable_legacy_bridge: Enable bridging to legacy components
            enable_event_migration: Enable migrating legacy events to Redis
            enable_real_time_sync: Enable real-time synchronization
        """
        self.enable_legacy_bridge = enable_legacy_bridge
        self.enable_event_migration = enable_event_migration
        self.enable_real_time_sync = enable_real_time_sync
        
        self.logger = logging.getLogger(f"{__name__}.RedisIntelligenceIntegrator")
        
        # Redis components
        self.intelligence_hub: Optional[RedisIntelligenceHub] = None
        self.event_store: Optional[RedisEventStore] = None
        self.workflow_engine: Optional[RedisWorkflowEngine] = None
        self.state_manager = None
        
        # Legacy components
        self.legacy_swarm: Optional['SwarmIntelligence'] = None
        self.legacy_hub: Optional['IntelligenceHub'] = None
        self.legacy_brain: Optional['AIBrain'] = None
        self.legacy_task_manager: Optional['TaskManager'] = None
        self.legacy_god_mode: Optional['GodModeController'] = None
        
        # Integration state
        self._integration_tasks: List[asyncio.Task] = []
        self._event_bridges: Dict[str, Callable] = {}
        self._sync_tasks: List[asyncio.Task] = []
        self._shutdown = False
        
        # Integration metrics
        self._metrics = {
            'events_bridged': 0,
            'legacy_events_migrated': 0,
            'sync_operations': 0,
            'integration_errors': 0,
            'components_integrated': 0
        }
    
    async def initialize(self):
        """Initialize integration components."""
        try:
            self.logger.info("Initializing Redis Intelligence Integrator")
            
            # Initialize Redis components
            from redis_intelligence_hub import get_intelligence_hub
            from redis_event_sourcing import get_event_store
            from redis_distributed_workflows import get_workflow_engine
            
            self.intelligence_hub = await get_intelligence_hub()
            self.event_store = await get_event_store()
            self.workflow_engine = await get_workflow_engine()
            self.state_manager = create_lockfree_state_manager(f"streams_integration_{self.integration_id}")
            await self.state_manager.initialize()
            
            # Initialize legacy components if available
            if LEGACY_COMPONENTS_AVAILABLE and self.enable_legacy_bridge:
                await self._initialize_legacy_components()
            
            # Set up event bridges
            await self._setup_event_bridges()
            
            # Start integration tasks
            await self._start_integration_tasks()
            
            self.logger.info("Redis Intelligence Integrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing integrator: {e}")
            raise
    
    async def _initialize_legacy_components(self):
        """Initialize legacy intelligence components."""
        try:
            self.logger.info("Initializing legacy component integration")
            
            # Initialize SwarmIntelligence integration
            try:
                self.legacy_swarm = SwarmIntelligence()
                await self._integrate_swarm_intelligence()
                self._metrics['components_integrated'] += 1
            except Exception as e:
                self.logger.warning(f"Could not integrate SwarmIntelligence: {e}")
            
            # Initialize IntelligenceHub integration
            try:
                self.legacy_hub = IntelligenceHub()
                await self._integrate_intelligence_hub()
                self._metrics['components_integrated'] += 1
            except Exception as e:
                self.logger.warning(f"Could not integrate IntelligenceHub: {e}")
            
            # Initialize AIBrain integration
            try:
                self.legacy_brain = AIBrain()
                await self._integrate_ai_brain()
                self._metrics['components_integrated'] += 1
            except Exception as e:
                self.logger.warning(f"Could not integrate AIBrain: {e}")
            
            # Initialize TaskManager integration
            try:
                self.legacy_task_manager = TaskManager()
                await self._integrate_task_manager()
                self._metrics['components_integrated'] += 1
            except Exception as e:
                self.logger.warning(f"Could not integrate TaskManager: {e}")
            
            # Initialize GodModeController integration
            try:
                self.legacy_god_mode = GodModeController()
                await self._integrate_god_mode_controller()
                self._metrics['components_integrated'] += 1
            except Exception as e:
                self.logger.warning(f"Could not integrate GodModeController: {e}")
            
            self.logger.info(f"Integrated {self._metrics['components_integrated']} legacy components")
            
        except Exception as e:
            self.logger.error(f"Error initializing legacy components: {e}")
    
    async def _integrate_swarm_intelligence(self):
        """Integrate SwarmIntelligence with Redis Streams."""
        try:
            # Bridge swarm events to Redis
            self._event_bridges['swarm_coordination'] = self._bridge_swarm_events
            
            # Register swarm event processors
            self.intelligence_hub.register_event_processor(
                EventType.COORDINATION_EVENT,
                self._handle_swarm_coordination_event
            )
            
            # Migrate existing swarm state
            if self.enable_event_migration and hasattr(self.legacy_swarm, 'get_swarm_state'):
                await self._migrate_swarm_state()
            
            self.logger.info("SwarmIntelligence integration completed")
            
        except Exception as e:
            self.logger.error(f"Error integrating SwarmIntelligence: {e}")
    
    async def _integrate_intelligence_hub(self):
        """Integrate legacy IntelligenceHub with Redis Streams."""
        try:
            # Bridge intelligence events to Redis
            self._event_bridges['intelligence_insights'] = self._bridge_intelligence_events
            
            # Register intelligence event processors
            self.intelligence_hub.register_event_processor(
                EventType.INTELLIGENCE_UPDATE,
                self._handle_intelligence_update_event
            )
            
            # Migrate existing intelligence data
            if self.enable_event_migration and hasattr(self.legacy_hub, 'get_intelligence_data'):
                await self._migrate_intelligence_data()
            
            self.logger.info("IntelligenceHub integration completed")
            
        except Exception as e:
            self.logger.error(f"Error integrating IntelligenceHub: {e}")
    
    async def _integrate_ai_brain(self):
        """Integrate AIBrain with Redis Streams."""
        try:
            # Bridge AI brain events to Redis
            self._event_bridges['ai_brain_decisions'] = self._bridge_ai_brain_events
            
            # Register AI brain event processors
            self.intelligence_hub.register_event_processor(
                EventType.AI_REQUEST,
                self._handle_ai_brain_request_event
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.AI_RESPONSE,
                self._handle_ai_brain_response_event
            )
            
            # Migrate existing AI brain data
            if self.enable_event_migration and hasattr(self.legacy_brain, 'get_brain_state'):
                await self._migrate_ai_brain_state()
            
            self.logger.info("AIBrain integration completed")
            
        except Exception as e:
            self.logger.error(f"Error integrating AIBrain: {e}")
    
    async def _integrate_task_manager(self):
        """Integrate TaskManager with Redis Streams."""
        try:
            # Bridge task events to Redis
            self._event_bridges['task_management'] = self._bridge_task_events
            
            # Register task event processors
            self.intelligence_hub.register_event_processor(
                EventType.TASK_ASSIGNMENT,
                self._handle_legacy_task_assignment
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.TASK_COMPLETION,
                self._handle_legacy_task_completion
            )
            
            # Migrate existing task data
            if self.enable_event_migration and hasattr(self.legacy_task_manager, 'get_all_tasks'):
                await self._migrate_task_data()
            
            self.logger.info("TaskManager integration completed")
            
        except Exception as e:
            self.logger.error(f"Error integrating TaskManager: {e}")
    
    async def _integrate_god_mode_controller(self):
        """Integrate GodModeController with Redis Streams."""
        try:
            # Bridge god mode events to Redis
            self._event_bridges['god_mode_control'] = self._bridge_god_mode_events
            
            # Register god mode event processors
            self.intelligence_hub.register_event_processor(
                EventType.COORDINATION_EVENT,
                self._handle_god_mode_coordination
            )
            
            # Migrate existing god mode state
            if self.enable_event_migration and hasattr(self.legacy_god_mode, 'get_control_state'):
                await self._migrate_god_mode_state()
            
            self.logger.info("GodModeController integration completed")
            
        except Exception as e:
            self.logger.error(f"Error integrating GodModeController: {e}")
    
    async def _setup_event_bridges(self):
        """Set up bidirectional event bridges between Redis and legacy systems."""
        try:
            # Register Redis event handlers for legacy system updates
            self.intelligence_hub.register_event_processor(
                EventType.WORKER_REGISTRATION,
                self._bridge_to_legacy_systems
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.TASK_COMPLETION,
                self._bridge_to_legacy_systems
            )
            
            self.intelligence_hub.register_event_processor(
                EventType.PERFORMANCE_METRIC,
                self._bridge_to_legacy_systems
            )
            
            self.logger.info("Event bridges configured")
            
        except Exception as e:
            self.logger.error(f"Error setting up event bridges: {e}")
    
    async def _start_integration_tasks(self):
        """Start background integration tasks."""
        try:
            # Start real-time sync task
            if self.enable_real_time_sync:
                sync_task = asyncio.create_task(self._real_time_sync_task())
                self._sync_tasks.append(sync_task)
            
            # Start legacy event monitoring
            if self.enable_legacy_bridge:
                monitor_task = asyncio.create_task(self._legacy_event_monitor())
                self._integration_tasks.append(monitor_task)
            
            # Start state synchronization
            state_sync_task = asyncio.create_task(self._state_synchronization_task())
            self._integration_tasks.append(state_sync_task)
            
            self.logger.info("Integration background tasks started")
            
        except Exception as e:
            self.logger.error(f"Error starting integration tasks: {e}")
    
    async def _bridge_swarm_events(self, legacy_event: Dict[str, Any]):
        """Bridge swarm intelligence events to Redis."""
        try:
            redis_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.COORDINATION_EVENT,
                worker_id=legacy_event.get('worker_id', 'swarm_legacy'),
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={
                    'event_subtype': 'swarm_coordination',
                    'swarm_data': legacy_event,
                    'source': 'legacy_swarm'
                }
            )
            
            await self.intelligence_hub.publish_event(redis_event)
            self._metrics['events_bridged'] += 1
            
        except Exception as e:
            self.logger.error(f"Error bridging swarm event: {e}")
            self._metrics['integration_errors'] += 1
    
    async def _bridge_intelligence_events(self, legacy_event: Dict[str, Any]):
        """Bridge intelligence hub events to Redis."""
        try:
            redis_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.INTELLIGENCE_UPDATE,
                worker_id=legacy_event.get('worker_id', 'intelligence_legacy'),
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={
                    'intelligence_data': legacy_event,
                    'source': 'legacy_intelligence_hub'
                }
            )
            
            await self.intelligence_hub.publish_event(redis_event)
            self._metrics['events_bridged'] += 1
            
        except Exception as e:
            self.logger.error(f"Error bridging intelligence event: {e}")
            self._metrics['integration_errors'] += 1
    
    async def _bridge_ai_brain_events(self, legacy_event: Dict[str, Any]):
        """Bridge AI brain events to Redis."""
        try:
            event_type = EventType.AI_REQUEST if legacy_event.get('type') == 'request' else EventType.AI_RESPONSE
            
            redis_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                worker_id=legacy_event.get('worker_id', 'ai_brain_legacy'),
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={
                    'ai_brain_data': legacy_event,
                    'source': 'legacy_ai_brain'
                }
            )
            
            await self.intelligence_hub.publish_event(redis_event)
            self._metrics['events_bridged'] += 1
            
        except Exception as e:
            self.logger.error(f"Error bridging AI brain event: {e}")
            self._metrics['integration_errors'] += 1
    
    async def _bridge_task_events(self, legacy_event: Dict[str, Any]):
        """Bridge task manager events to Redis."""
        try:
            event_type_map = {
                'assignment': EventType.TASK_ASSIGNMENT,
                'completion': EventType.TASK_COMPLETION,
                'failure': EventType.TASK_FAILURE,
                'progress': EventType.TASK_PROGRESS
            }
            
            event_type = event_type_map.get(legacy_event.get('type'), EventType.TASK_ASSIGNMENT)
            
            redis_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                worker_id=legacy_event.get('worker_id', 'task_manager_legacy'),
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.NORMAL,
                data={
                    'task_data': legacy_event,
                    'source': 'legacy_task_manager'
                }
            )
            
            await self.intelligence_hub.publish_event(redis_event)
            self._metrics['events_bridged'] += 1
            
        except Exception as e:
            self.logger.error(f"Error bridging task event: {e}")
            self._metrics['integration_errors'] += 1
    
    async def _bridge_god_mode_events(self, legacy_event: Dict[str, Any]):
        """Bridge god mode controller events to Redis."""
        try:
            redis_event = IntelligenceEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.COORDINATION_EVENT,
                worker_id=legacy_event.get('worker_id', 'god_mode_legacy'),
                timestamp=datetime.now(timezone.utc),
                priority=EventPriority.HIGH,
                data={
                    'event_subtype': 'god_mode_control',
                    'control_data': legacy_event,
                    'source': 'legacy_god_mode'
                }
            )
            
            await self.intelligence_hub.publish_event(redis_event)
            self._metrics['events_bridged'] += 1
            
        except Exception as e:
            self.logger.error(f"Error bridging god mode event: {e}")
            self._metrics['integration_errors'] += 1
    
    async def _bridge_to_legacy_systems(self, event: IntelligenceEvent, stream_name: str):
        """Bridge Redis events back to legacy systems."""
        try:
            # Convert Redis event to legacy format
            legacy_event = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'worker_id': event.worker_id,
                'timestamp': event.timestamp.isoformat(),
                'data': event.data,
                'source': 'redis_streams'
            }
            
            # Send to appropriate legacy systems
            if self.legacy_swarm and event.event_type == EventType.COORDINATION_EVENT:
                await self._send_to_swarm_intelligence(legacy_event)
            
            if self.legacy_hub and event.event_type == EventType.INTELLIGENCE_UPDATE:
                await self._send_to_intelligence_hub(legacy_event)
            
            if self.legacy_brain and event.event_type in [EventType.AI_REQUEST, EventType.AI_RESPONSE]:
                await self._send_to_ai_brain(legacy_event)
            
            if self.legacy_task_manager and event.event_type in [
                EventType.TASK_ASSIGNMENT, EventType.TASK_COMPLETION, EventType.TASK_FAILURE
            ]:
                await self._send_to_task_manager(legacy_event)
            
            if self.legacy_god_mode and event.event_type == EventType.COORDINATION_EVENT:
                await self._send_to_god_mode(legacy_event)
            
        except Exception as e:
            self.logger.error(f"Error bridging to legacy systems: {e}")
            self._metrics['integration_errors'] += 1
    
    async def _send_to_swarm_intelligence(self, event: Dict[str, Any]):
        """Send event to legacy SwarmIntelligence."""
        try:
            if hasattr(self.legacy_swarm, 'handle_redis_event'):
                await self.legacy_swarm.handle_redis_event(event)
            elif hasattr(self.legacy_swarm, 'process_coordination_event'):
                await self.legacy_swarm.process_coordination_event(event)
        except Exception as e:
            self.logger.debug(f"Could not send event to SwarmIntelligence: {e}")
    
    async def _send_to_intelligence_hub(self, event: Dict[str, Any]):
        """Send event to legacy IntelligenceHub."""
        try:
            if hasattr(self.legacy_hub, 'handle_redis_event'):
                await self.legacy_hub.handle_redis_event(event)
            elif hasattr(self.legacy_hub, 'process_intelligence_update'):
                await self.legacy_hub.process_intelligence_update(event)
        except Exception as e:
            self.logger.debug(f"Could not send event to IntelligenceHub: {e}")
    
    async def _send_to_ai_brain(self, event: Dict[str, Any]):
        """Send event to legacy AIBrain."""
        try:
            if hasattr(self.legacy_brain, 'handle_redis_event'):
                await self.legacy_brain.handle_redis_event(event)
            elif hasattr(self.legacy_brain, 'process_ai_event'):
                await self.legacy_brain.process_ai_event(event)
        except Exception as e:
            self.logger.debug(f"Could not send event to AIBrain: {e}")
    
    async def _send_to_task_manager(self, event: Dict[str, Any]):
        """Send event to legacy TaskManager."""
        try:
            if hasattr(self.legacy_task_manager, 'handle_redis_event'):
                await self.legacy_task_manager.handle_redis_event(event)
            elif hasattr(self.legacy_task_manager, 'process_task_event'):
                await self.legacy_task_manager.process_task_event(event)
        except Exception as e:
            self.logger.debug(f"Could not send event to TaskManager: {e}")
    
    async def _send_to_god_mode(self, event: Dict[str, Any]):
        """Send event to legacy GodModeController."""
        try:
            if hasattr(self.legacy_god_mode, 'handle_redis_event'):
                await self.legacy_god_mode.handle_redis_event(event)
            elif hasattr(self.legacy_god_mode, 'process_control_event'):
                await self.legacy_god_mode.process_control_event(event)
        except Exception as e:
            self.logger.debug(f"Could not send event to GodModeController: {e}")
    
    async def _real_time_sync_task(self):
        """Real-time synchronization between Redis and legacy systems."""
        while not self._shutdown:
            try:
                # Sync worker registrations
                await self._sync_worker_registrations()
                
                # Sync task states
                await self._sync_task_states()
                
                # Sync performance metrics
                await self._sync_performance_metrics()
                
                self._metrics['sync_operations'] += 1
                
                await asyncio.sleep(10)  # Sync every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in real-time sync: {e}")
                await asyncio.sleep(30)
    
    async def _legacy_event_monitor(self):
        """Monitor legacy systems for events to bridge."""
        while not self._shutdown:
            try:
                # Monitor each legacy component for events
                for bridge_name, bridge_func in self._event_bridges.items():
                    # This would need to be implemented based on each legacy system's event mechanism
                    # For now, we'll use a polling approach
                    pass
                
                await asyncio.sleep(1)  # Monitor every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in legacy event monitor: {e}")
                await asyncio.sleep(5)
    
    async def _state_synchronization_task(self):
        """Synchronize state between Redis and legacy systems."""
        while not self._shutdown:
            try:
                # Sync global state
                await self._sync_global_state()
                
                # Sync component-specific states
                await self._sync_component_states()
                
                await asyncio.sleep(60)  # Sync every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in state synchronization: {e}")
                await asyncio.sleep(120)
    
    async def _migrate_swarm_state(self):
        """Migrate existing swarm state to Redis."""
        try:
            if hasattr(self.legacy_swarm, 'get_swarm_state'):
                swarm_state = await self.legacy_swarm.get_swarm_state()
                
                # Convert to Redis events
                migration_event = IntelligenceEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType.COORDINATION_EVENT,
                    worker_id='migration_service',
                    timestamp=datetime.now(timezone.utc),
                    priority=EventPriority.LOW,
                    data={
                        'event_subtype': 'swarm_state_migration',
                        'swarm_state': swarm_state,
                        'migration_timestamp': datetime.now(timezone.utc).isoformat()
                    }
                )
                
                await self.event_store.append_event(migration_event)
                self._metrics['legacy_events_migrated'] += 1
                
                self.logger.info("Swarm state migrated to Redis")
        
        except Exception as e:
            self.logger.error(f"Error migrating swarm state: {e}")
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status."""
        return {
            'integrator_status': 'active' if not self._shutdown else 'shutdown',
            'legacy_bridge_enabled': self.enable_legacy_bridge,
            'event_migration_enabled': self.enable_event_migration,
            'real_time_sync_enabled': self.enable_real_time_sync,
            'components_integrated': self._metrics['components_integrated'],
            'legacy_components_available': LEGACY_COMPONENTS_AVAILABLE,
            'integrated_components': {
                'swarm_intelligence': self.legacy_swarm is not None,
                'intelligence_hub': self.legacy_hub is not None,
                'ai_brain': self.legacy_brain is not None,
                'task_manager': self.legacy_task_manager is not None,
                'god_mode_controller': self.legacy_god_mode is not None
            },
            'event_bridges_configured': len(self._event_bridges),
            'metrics': self._metrics.copy()
        }
    
    async def shutdown(self):
        """Shutdown integration services."""
        self.logger.info("Shutting down Redis Intelligence Integrator")
        self._shutdown = True
        
        # Stop integration tasks
        all_tasks = self._integration_tasks + self._sync_tasks
        
        for task in all_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Redis Intelligence Integrator shutdown complete")


# Global integrator instance
_global_integrator: Optional[RedisIntelligenceIntegrator] = None


async def get_intelligence_integrator(**kwargs) -> RedisIntelligenceIntegrator:
    """Get global intelligence integrator instance."""
    global _global_integrator
    
    if _global_integrator is None:
        _global_integrator = RedisIntelligenceIntegrator(**kwargs)
        await _global_integrator.initialize()
    
    return _global_integrator


async def create_intelligence_integrator(**kwargs) -> RedisIntelligenceIntegrator:
    """Create new intelligence integrator instance."""
    integrator = RedisIntelligenceIntegrator(**kwargs)
    await integrator.initialize()
    return integrator


# Enhanced worker intelligence adapter
class EnhancedWorkerIntelligenceAdapter:
    """Adapter that enhances existing worker intelligence with Redis Streams."""
    
    def __init__(self, worker_id: str):
        """Initialize enhanced worker intelligence adapter.
        
        Args:
            worker_id: Unique worker identifier
        """
        self.worker_id = worker_id
        self.logger = logging.getLogger(f"{__name__}.EnhancedWorkerIntelligenceAdapter")
        
        # Redis components
        self.intelligence_hub: Optional[RedisIntelligenceHub] = None
        self.event_store: Optional[RedisEventStore] = None
        self.integrator: Optional[RedisIntelligenceIntegrator] = None
        
        # Enhanced capabilities
        self._capabilities = []
        self._performance_metrics = {}
        self._task_history = []
        self._intelligence_data = {}
    
    async def initialize(self):
        """Initialize enhanced worker intelligence."""
        try:
            # Get Redis components
            self.intelligence_hub = await get_intelligence_hub()
            self.event_store = await get_event_store()
            self.integrator = await get_intelligence_integrator()
            
            # Register worker
            await self.register_worker()
            
            self.logger.info(f"Enhanced worker intelligence initialized for {self.worker_id}")
            
        except Exception as e:
            self.logger.error(f"Error initializing enhanced worker intelligence: {e}")
            raise
    
    async def register_worker(self, capabilities: List[str] = None):
        """Register worker with enhanced intelligence hub."""
        self._capabilities = capabilities or []
        
        await self.intelligence_hub.register_worker(self.worker_id, self._capabilities)
        
        self.logger.info(f"Worker {self.worker_id} registered with capabilities: {self._capabilities}")
    
    async def send_heartbeat(self, performance_data: Dict[str, Any] = None):
        """Send enhanced heartbeat with performance data."""
        if performance_data:
            self._performance_metrics.update(performance_data)
        
        await self.intelligence_hub.worker_heartbeat(self.worker_id, self._performance_metrics)
    
    async def report_task_completion(self, task_id: str, result: Dict[str, Any], duration: float):
        """Report enhanced task completion."""
        await self.intelligence_hub.report_task_completion(
            self.worker_id, task_id, result, duration
        )
        
        # Add to task history
        self._task_history.append({
            'task_id': task_id,
            'completion_time': datetime.now(timezone.utc).isoformat(),
            'duration': duration,
            'success': True
        })
    
    async def update_intelligence(self, intelligence_data: Dict[str, Any]):
        """Update worker intelligence data."""
        self._intelligence_data.update(intelligence_data)
        
        # Publish intelligence update event
        event = IntelligenceEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.INTELLIGENCE_UPDATE,
            worker_id=self.worker_id,
            timestamp=datetime.now(timezone.utc),
            priority=EventPriority.NORMAL,
            data=intelligence_data
        )
        
        await self.intelligence_hub.publish_event(event)
    
    async def get_worker_analytics(self) -> Dict[str, Any]:
        """Get comprehensive worker analytics."""
        return {
            'worker_id': self.worker_id,
            'capabilities': self._capabilities,
            'performance_metrics': self._performance_metrics,
            'task_history_count': len(self._task_history),
            'intelligence_data': self._intelligence_data,
            'redis_integration_active': self.intelligence_hub is not None
        }


async def enhance_existing_worker(worker_id: str, capabilities: List[str] = None) -> EnhancedWorkerIntelligenceAdapter:
    """Enhance existing worker with Redis Streams intelligence.
    
    Args:
        worker_id: Worker to enhance
        capabilities: Worker capabilities
        
    Returns:
        Enhanced worker adapter
    """
    adapter = EnhancedWorkerIntelligenceAdapter(worker_id)
    await adapter.initialize()
    
    if capabilities:
        await adapter.register_worker(capabilities)
    
    return adapter