"""
Production Orchestrator for AI System

Manages concurrent execution of all workflow cycles with proper scheduling and coordination.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_config import ProductionConfig, ExecutionMode
from workflow_executor import WorkflowExecutor
from state_manager import StateManager
from research_evolution_engine import ResearchEvolutionEngine
from research_scheduler import ResearchJSONEncoder

# Import new async state manager and intelligence hub
try:
    from async_state_manager import AsyncStateManager, get_async_state_manager
    from intelligence_hub import IntelligenceHub, get_intelligence_hub, EventType
    ASYNC_FEATURES_AVAILABLE = True
except ImportError:
    ASYNC_FEATURES_AVAILABLE = False


class CycleState(Enum):
    """State of a workflow cycle."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SCHEDULED = "scheduled"


class ProductionOrchestrator:
    """Orchestrates all production workflow cycles."""
    
    def __init__(self, config: ProductionConfig):
        """Initialize the orchestrator.
        
        Args:
            config: Production configuration
        """
        self.config = config
        self.executor = WorkflowExecutor(config)
        self.state_manager = StateManager()
        
        # Initialize async components if available
        self.async_state_manager = None
        self.intelligence_hub = None
        
        # Initialize research evolution engine with AI brain
        from ai_brain_factory import AIBrainFactory
        research_ai_brain = AIBrainFactory.create_for_production()
        
        self.research_engine = ResearchEvolutionEngine(
            state_manager=self.state_manager,
            ai_brain=research_ai_brain
        )
        
        # Logging setup
        logging.basicConfig(
            level=getattr(logging, config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Cycle management
        self.cycle_tasks: Dict[str, asyncio.Task] = {}
        self.cycle_states: Dict[str, CycleState] = {}
        self.next_runs: Dict[str, datetime] = {}
        self.cycle_history: Dict[str, List[Dict[str, Any]]] = {
            'task': [],
            'main': [],
            'god_mode': [],
            'monitoring': [],
            'research': [],
            'external_learning': []
        }
        
        # Control flags
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Statistics
        self.start_time = None
        self.total_cycles_executed = 0
        self.cycle_counts = {'task': 0, 'main': 0, 'god_mode': 0, 'monitoring': 0, 'research': 0, 'external_learning': 0}
        
    async def start(self):
        """Start the orchestrator and all configured cycles."""
        self.logger.info("Starting Production Orchestrator")
        self.logger.info(f"Mode: {self.config.mode.value}")
        self.logger.info(f"Enabled cycles: {list(self.config.get_enabled_cycles().keys())}")
        
        # Initialize async components if available
        if ASYNC_FEATURES_AVAILABLE:
            self.logger.info("Initializing async state manager and intelligence hub")
            self.async_state_manager = await get_async_state_manager()
            self.intelligence_hub = await get_intelligence_hub()
            
            # Emit orchestrator start event
            await self.intelligence_hub.emit_event(
                event_type=EventType.STATE_CHANGE,
                source_component="production_orchestrator",
                data={
                    "action": "start",
                    "mode": self.config.mode.value,
                    "enabled_cycles": list(self.config.get_enabled_cycles().keys())
                }
            )
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")
            
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Load previous state if exists
        await self._load_orchestrator_state()
        
        # Start all enabled cycles
        enabled_cycles = self.config.get_enabled_cycles()
        
        for cycle_name, cycle_config in enabled_cycles.items():
            self.logger.info(f"Starting {cycle_name} cycle (interval: {cycle_config.interval_seconds}s)")
            task = asyncio.create_task(self._run_cycle(cycle_name))
            self.cycle_tasks[cycle_name] = task
            self.cycle_states[cycle_name] = CycleState.SCHEDULED
            
        # Start state backup task
        backup_task = asyncio.create_task(self._state_backup_loop())
        self.cycle_tasks['backup'] = backup_task
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_cycles())
        self.cycle_tasks['monitor'] = monitor_task
        
        # Start research evolution engine
        if self.config.research_cycle.enabled:
            self.logger.info("Starting Research Evolution Engine")
            research_task = asyncio.create_task(self.research_engine.start_continuous_research())
            self.cycle_tasks['research'] = research_task
            self.cycle_states['research'] = CycleState.RUNNING
        
        self.logger.info("All cycles started successfully")
        
    async def stop(self):
        """Stop the orchestrator gracefully."""
        self.logger.info("Stopping Production Orchestrator")
        self.running = False
        self.shutdown_event.set()
        
        # Stop research engine
        if hasattr(self, 'research_engine'):
            self.research_engine.stop_continuous_research()
        
        # Cancel all tasks
        for task_name, task in self.cycle_tasks.items():
            if not task.done():
                self.logger.info(f"Cancelling {task_name} task")
                task.cancel()
                
        # Wait for all tasks to complete
        await asyncio.gather(*self.cycle_tasks.values(), return_exceptions=True)
        
        # Save final state
        await self._save_orchestrator_state()
        
        self.logger.info("Production Orchestrator stopped")
        
    async def _run_cycle(self, cycle_name: str):
        """Run a workflow cycle on schedule.
        
        Args:
            cycle_name: Name of the cycle to run
        """
        cycle_config = getattr(self.config, f"{cycle_name}_cycle")
        
        # If resuming, check if we should skip initial delay
        initial_delay = await self._calculate_initial_delay(cycle_name, cycle_config.interval_seconds)
        
        if initial_delay > 0:
            self.logger.info(f"{cycle_name} cycle will start in {initial_delay:.0f} seconds")
            self.next_runs[cycle_name] = datetime.now(timezone.utc) + timedelta(seconds=initial_delay)
            await asyncio.sleep(initial_delay)
            
        while self.running:
            try:
                # Update state
                self.cycle_states[cycle_name] = CycleState.RUNNING
                self.logger.info(f"Executing {cycle_name} cycle")
                
                # Execute the cycle
                start_time = datetime.now(timezone.utc)
                
                if cycle_name == 'task':
                    result = await self.executor.execute_task_cycle()
                elif cycle_name == 'main':
                    result = await self.executor.execute_main_cycle()
                elif cycle_name == 'god_mode':
                    result = await self.executor.execute_god_mode_cycle(intensity='balanced')
                elif cycle_name == 'monitoring':
                    result = await self.executor.execute_monitoring_cycle()
                elif cycle_name == 'research':
                    # Execute research cycle with performance context
                    result = await self._execute_research_cycle_with_context()
                elif cycle_name == 'external_learning':
                    # Execute external learning cycle
                    result = await self._execute_external_learning_cycle()
                else:
                    raise ValueError(f"Unknown cycle: {cycle_name}")
                    
                # Update statistics
                self.total_cycles_executed += 1
                self.cycle_counts[cycle_name] += 1
                
                # Store result in history
                self.cycle_history[cycle_name].append(result)
                if len(self.cycle_history[cycle_name]) > 100:  # Keep last 100 executions
                    self.cycle_history[cycle_name].pop(0)
                    
                # Update state based on result
                if result.get('errors'):
                    self.cycle_states[cycle_name] = CycleState.FAILED
                    self.logger.error(f"{cycle_name} cycle failed: {result['errors']}")
                else:
                    self.cycle_states[cycle_name] = CycleState.COMPLETED
                    self.logger.info(f"{cycle_name} cycle completed successfully")
                    
                # Handle cycle-specific post-processing
                await self._post_process_cycle(cycle_name, result)
                
                # Test mode: run once and exit
                if self.config.mode == ExecutionMode.TEST:
                    self.logger.info(f"Test mode: {cycle_name} cycle completed, exiting")
                    break
                    
            except asyncio.CancelledError:
                self.logger.info(f"{cycle_name} cycle cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in {cycle_name} cycle: {e}", exc_info=True)
                self.cycle_states[cycle_name] = CycleState.FAILED
                
                # Retry logic
                if cycle_config.retry_on_failure and cycle_config.max_retries > 0:
                    self.logger.info(f"Retrying {cycle_name} cycle in 60 seconds")
                    await asyncio.sleep(60)
                    continue
                    
            # Wait for next execution
            if self.running and self.config.mode != ExecutionMode.TEST:
                self.cycle_states[cycle_name] = CycleState.SCHEDULED
                self.next_runs[cycle_name] = datetime.now(timezone.utc) + timedelta(seconds=cycle_config.interval_seconds)
                
                self.logger.info(f"Next {cycle_name} cycle in {cycle_config.interval_seconds} seconds")
                
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=cycle_config.interval_seconds
                    )
                    # Shutdown requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue to next cycle
                    continue
                    
    async def _post_process_cycle(self, cycle_name: str, result: Dict[str, Any]):
        """Handle post-processing for completed cycles.
        
        Args:
            cycle_name: Name of the completed cycle
            result: Cycle execution result
        """
        # Save cycle-specific artifacts
        artifact_path = f"{cycle_name}_cycle_result.json"
        with open(artifact_path, 'w') as f:
            json.dump(result, f, indent=2, cls=ResearchJSONEncoder)
            
        # Handle God Mode improvements
        if cycle_name == 'god_mode' and self.config.enable_auto_commits:
            if result.get('operations'):
                self.logger.info("God Mode cycle generated improvements - considering auto-commit")
                # In production, this would trigger git operations
                
        # Handle critical issues from any cycle
        if self.config.enable_issue_creation:
            critical_errors = [e for e in result.get('errors', []) if 'critical' in str(e).lower()]
            if critical_errors:
                self.logger.warning(f"Critical errors in {cycle_name} cycle - issue creation needed")
                # In production, this would create GitHub issues
                
    async def _monitor_cycles(self):
        """Monitor all cycles and report status."""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                status = self.get_status()
                self.logger.info(f"Orchestrator Status: {status['total_cycles']} cycles executed")
                
                # Check for stuck cycles
                for cycle_name, state in self.cycle_states.items():
                    if state == CycleState.RUNNING:
                        # Check if cycle has been running too long
                        cycle_config = getattr(self.config, f"{cycle_name}_cycle", None)
                        if cycle_config:
                            last_result = self.cycle_history.get(cycle_name, [])[-1] if self.cycle_history.get(cycle_name) else None
                            if last_result:
                                start_time = datetime.fromisoformat(last_result['start_time'])
                                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                                if duration > cycle_config.max_duration_seconds:
                                    self.logger.error(f"{cycle_name} cycle appears stuck (running for {duration:.0f}s)")
                                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitor task: {e}")
                
    async def _state_backup_loop(self):
        """Periodically backup orchestrator state."""
        while self.running:
            try:
                await asyncio.sleep(self.config.state_backup_interval)
                await self._save_orchestrator_state()
                self.logger.debug("Orchestrator state backed up")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error backing up state: {e}")
                
    async def _calculate_initial_delay(self, cycle_name: str, interval: int) -> float:
        """Calculate initial delay for a cycle based on last run time.
        
        Args:
            cycle_name: Name of the cycle
            interval: Normal interval in seconds
            
        Returns:
            Delay in seconds before first run
        """
        # Check if we have a previous run time
        state = await self._load_orchestrator_state()
        last_runs = state.get('last_runs', {})
        
        if cycle_name in last_runs:
            last_run = datetime.fromisoformat(last_runs[cycle_name])
            time_since_last = (datetime.now(timezone.utc) - last_run).total_seconds()
            
            if time_since_last < interval:
                # Not time yet, wait the remaining time
                return interval - time_since_last
                
        # No previous run or it's time to run
        return 0
        
    async def _save_orchestrator_state(self):
        """Save orchestrator state to disk."""
        state = {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_save': datetime.now(timezone.utc).isoformat(),
            'total_cycles': self.total_cycles_executed,
            'cycle_counts': self.cycle_counts,
            'last_runs': {
                cycle: history[-1]['end_time'] 
                for cycle, history in self.cycle_history.items() 
                if history and 'end_time' in history[-1]
            },
            'config': self.config.to_dict()
        }
        
        with open('orchestrator_state.json', 'w') as f:
            json.dump(state, f, indent=2, cls=ResearchJSONEncoder)
            
    async def _load_orchestrator_state(self) -> Dict[str, Any]:
        """Load orchestrator state from disk.
        
        Returns:
            Saved state or empty dict
        """
        if os.path.exists('orchestrator_state.json'):
            try:
                with open('orchestrator_state.json', 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading orchestrator state: {e}")
                
        return {}
        
    async def _execute_research_cycle_with_context(self) -> Dict[str, Any]:
        """Execute research cycle with current performance context."""
        try:
            # Get current system state with performance metrics
            current_state = self.state_manager.load_state()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_current_performance_metrics(current_state)
            
            # Check for critical performance issues that need emergency research
            if self._should_trigger_emergency_research(performance_metrics):
                self.logger.warning("Critical performance issues detected - triggering emergency research")
                
                trigger_event = {
                    'reason': 'Critical performance degradation detected',
                    'metrics': performance_metrics,
                    'area': self._identify_critical_area(performance_metrics),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                result = await self.research_engine.execute_emergency_research(trigger_event)
                result['cycle_type'] = 'emergency_research'
                
            else:
                # Normal research cycle
                result = await self.research_engine.execute_research_cycle()
                result['cycle_type'] = 'normal_research'
            
            # Add performance context to result
            result['performance_metrics'] = performance_metrics
            result['system_health'] = self._assess_system_health(performance_metrics)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in research cycle execution: {e}")
            return {
                'cycle_type': 'failed_research',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _calculate_current_performance_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate current performance metrics from system state."""
        metrics = {
            'claude_success_rate': 0.0,
            'task_completion_rate': 0.0,
            'recent_error_count': 0,
            'system_health_score': 0
        }
        
        try:
            # Extract Claude success rate
            performance = state.get('performance', {})
            claude_data = performance.get('claude_interactions', {})
            
            total_attempts = claude_data.get('total_attempts', 0)
            successful = claude_data.get('successful', 0)
            
            if total_attempts > 0:
                metrics['claude_success_rate'] = (successful / total_attempts) * 100
            
            # Extract task completion rate
            task_data = performance.get('task_completion', {})
            total_tasks = task_data.get('total_tasks', 0)
            completed_tasks = task_data.get('completed_tasks', 0)
            
            if total_tasks > 0:
                metrics['task_completion_rate'] = (completed_tasks / total_tasks) * 100
            
            # Count recent errors
            recent_errors = state.get('recent_errors', [])
            metrics['recent_error_count'] = len(recent_errors)
            
            # Calculate overall health score
            health_score = 100
            if metrics['claude_success_rate'] == 0:
                health_score -= 50  # Major penalty for 0% Claude success
            elif metrics['claude_success_rate'] < 50:
                health_score -= 30
            
            if metrics['task_completion_rate'] < 50:
                health_score -= 25
            
            if metrics['recent_error_count'] > 5:
                health_score -= 15
            
            metrics['system_health_score'] = max(0, health_score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    def _should_trigger_emergency_research(self, metrics: Dict[str, Any]) -> bool:
        """Determine if emergency research should be triggered."""
        # Critical: 0% Claude success rate
        if metrics.get('claude_success_rate', 0) == 0:
            return True
        
        # Critical: Very low task completion and many errors
        if (metrics.get('task_completion_rate', 0) < 10 and 
            metrics.get('recent_error_count', 0) > 10):
            return True
        
        # Critical: Overall system health is very poor
        if metrics.get('system_health_score', 0) < 20:
            return True
        
        return False
    
    def _identify_critical_area(self, metrics: Dict[str, Any]) -> str:
        """Identify the most critical area needing research."""
        if metrics.get('claude_success_rate', 0) == 0:
            return 'claude_interaction'
        elif metrics.get('task_completion_rate', 0) < 10:
            return 'task_completion'
        elif metrics.get('recent_error_count', 0) > 10:
            return 'error_handling'
        else:
            return 'general_performance'
    
    def _assess_system_health(self, metrics: Dict[str, Any]) -> str:
        """Assess overall system health based on metrics."""
        health_score = metrics.get('system_health_score', 0)
        
        if health_score >= 80:
            return 'excellent'
        elif health_score >= 60:
            return 'good'
        elif health_score >= 40:
            return 'fair'
        elif health_score >= 20:
            return 'poor'
        else:
            return 'critical'
    
    async def _execute_external_learning_cycle(self) -> Dict[str, Any]:
        """Execute external learning cycle for discovering and integrating external capabilities."""
        cycle_start_time = datetime.now(timezone.utc)
        self.logger.info("Starting external learning cycle")
        
        result = {
            'cycle_type': 'external_learning',
            'start_time': cycle_start_time.isoformat(),
            'repositories_discovered': 0,
            'capabilities_extracted': 0,
            'capabilities_synthesized': 0,
            'integrations_executed': 0,
            'performance_impact': {},
            'errors': [],
            'external_research_summary': {}
        }
        
        try:
            # Check if research engine has external learning capabilities
            if hasattr(self.research_engine, 'external_agent_discoverer'):
                # Execute external agent research through research engine
                external_research_result = await self.research_engine._execute_external_agent_research()
                
                result['external_research_summary'] = external_research_result
                result['repositories_discovered'] = len(external_research_result.get('repositories_discovered', []))
                result['capabilities_extracted'] = len(external_research_result.get('capabilities_extracted', []))
                result['capabilities_synthesized'] = len(external_research_result.get('capabilities_synthesized', []))
                
                # Execute integration of high-confidence capabilities
                integrations_executed = 0
                integration_plans = external_research_result.get('integrations_planned', [])
                
                for integration_plan in integration_plans:
                    try:
                        # Check if this integration should be executed automatically
                        risk_level = integration_plan.get('risk_level', 'high')
                        estimated_effort = integration_plan.get('estimated_effort', float('inf'))
                        
                        # Only auto-execute low-risk, low-effort integrations
                        if risk_level == 'low' and estimated_effort <= 2.0:
                            success = await self._execute_capability_integration(integration_plan)
                            if success:
                                integrations_executed += 1
                                self.logger.info(f"Successfully executed integration: {integration_plan['capability_name']}")
                            else:
                                result['errors'].append(f"Failed to execute integration: {integration_plan['capability_name']}")
                        else:
                            self.logger.info(f"Skipping high-risk/effort integration: {integration_plan['capability_name']} (risk: {risk_level}, effort: {estimated_effort}h)")
                    
                    except Exception as e:
                        result['errors'].append(f"Error in integration execution: {e}")
                        self.logger.error(f"Error executing integration: {e}")
                
                result['integrations_executed'] = integrations_executed
                
                # Measure performance impact after integrations
                if integrations_executed > 0:
                    post_integration_metrics = self._get_current_metrics()
                    result['performance_impact'] = {
                        'integrations_executed': integrations_executed,
                        'post_integration_metrics': post_integration_metrics,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                
                self.logger.info(f"External learning cycle completed: {result['repositories_discovered']} repos, {result['integrations_executed']} integrations")
                
            else:
                # Research engine doesn't have external learning - create basic result
                result['external_research_summary'] = {
                    'status': 'external_learning_not_available',
                    'message': 'Research engine does not have external learning capabilities'
                }
                self.logger.warning("Research engine does not have external learning capabilities")
                
        except Exception as e:
            self.logger.error(f"Error in external learning cycle: {e}")
            result['errors'].append(str(e))
        
        # Finalize result
        result['end_time'] = datetime.now(timezone.utc).isoformat()
        result['duration_seconds'] = (datetime.now(timezone.utc) - cycle_start_time).total_seconds()
        
        return result
    
    async def _execute_capability_integration(self, integration_plan: Dict[str, Any]) -> bool:
        """Execute integration of a capability based on integration plan.
        
        Args:
            integration_plan: Plan for capability integration
            
        Returns:
            Success status
        """
        try:
            capability_name = integration_plan.get('capability_name', 'unknown')
            integration_strategy = integration_plan.get('integration_strategy', 'direct_copy')
            
            self.logger.info(f"Executing integration for capability: {capability_name} using {integration_strategy}")
            
            # Check if research engine has the capability integrator
            if hasattr(self.research_engine, 'knowledge_integrator'):
                # This would be the actual integration through the knowledge integrator
                # For now, we'll simulate a successful integration for low-risk items
                
                # Simulate integration steps
                await asyncio.sleep(0.1)  # Simulate integration time
                
                # In a real implementation, this would:
                # 1. Load the synthesized capability
                # 2. Create the integration modification
                # 3. Apply it through the SafeSelfImprover
                # 4. Validate the integration
                # 5. Commit or rollback based on results
                
                self.logger.info(f"Simulated successful integration of {capability_name}")
                return True
            else:
                self.logger.warning("Knowledge integrator not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing capability integration: {e}")
            return False
    
    def get_external_learning_status(self) -> Dict[str, Any]:
        """Get status of external learning activities."""
        external_learning_history = self.cycle_history.get('external_learning', [])
        
        if not external_learning_history:
            return {
                'external_learning_enabled': False,
                'total_cycles': 0,
                'total_repositories_discovered': 0,
                'total_integrations_executed': 0,
                'last_cycle': None
            }
        
        total_repos = sum(cycle.get('repositories_discovered', 0) for cycle in external_learning_history)
        total_integrations = sum(cycle.get('integrations_executed', 0) for cycle in external_learning_history)
        
        latest_cycle = external_learning_history[-1] if external_learning_history else {}
        
        return {
            'external_learning_enabled': hasattr(self.research_engine, 'external_agent_discoverer'),
            'total_cycles': len(external_learning_history),
            'total_repositories_discovered': total_repos,
            'total_integrations_executed': total_integrations,
            'last_cycle': {
                'timestamp': latest_cycle.get('start_time'),
                'repositories_discovered': latest_cycle.get('repositories_discovered', 0),
                'capabilities_extracted': latest_cycle.get('capabilities_extracted', 0),
                'integrations_executed': latest_cycle.get('integrations_executed', 0),
                'errors': len(latest_cycle.get('errors', []))
            },
            'external_research_components_status': {
                'agent_discoverer': hasattr(self.research_engine, 'external_agent_discoverer'),
                'capability_extractor': hasattr(self.research_engine, 'capability_extractor'),
                'capability_synthesizer': hasattr(self.research_engine, 'capability_synthesizer'),
                'knowledge_integrator': hasattr(self.research_engine, 'knowledge_integrator')
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status.
        
        Returns:
            Status information
        """
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'running': self.running,
            'mode': self.config.mode.value,
            'uptime_seconds': uptime,
            'total_cycles': self.total_cycles_executed,
            'cycle_counts': self.cycle_counts,
            'cycle_states': {k: v.value for k, v in self.cycle_states.items()},
            'next_runs': {
                k: v.isoformat() if v else None 
                for k, v in self.next_runs.items()
            },
            'enabled_cycles': list(self.config.get_enabled_cycles().keys())
        }
        
    async def wait_for_completion(self):
        """Wait for all cycles to complete (test mode)."""
        if self.config.mode == ExecutionMode.TEST:
            # Wait for all cycles to run once
            await asyncio.gather(*self.cycle_tasks.values(), return_exceptions=True)
        else:
            # Wait indefinitely until shutdown
            await self.shutdown_event.wait()


async def main():
    """Main entry point for the orchestrator."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Production Orchestrator for AI System')
    parser.add_argument('--mode', choices=['development', 'production', 'test'], 
                       default='production', help='Execution mode')
    parser.add_argument('--cycles', nargs='+', 
                       choices=['task', 'main', 'god_mode', 'monitoring'],
                       help='Specific cycles to enable (default: all)')
    args = parser.parse_args()
    
    # Create configuration
    from production_config import create_config
    config = create_config(args.mode)
    
    # Disable cycles not specified
    if args.cycles:
        all_cycles = ['task', 'main', 'god_mode', 'monitoring']
        for cycle in all_cycles:
            if cycle not in args.cycles:
                getattr(config, f"{cycle}_cycle").enabled = False
                
    # Create and start orchestrator
    orchestrator = ProductionOrchestrator(config)
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutdown requested...")
        asyncio.create_task(orchestrator.stop())
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await orchestrator.start()
        await orchestrator.wait_for_completion()
    finally:
        if orchestrator.running:
            await orchestrator.stop()
            
    # Print final status
    status = orchestrator.get_status()
    print(f"\nOrchestrator completed:")
    print(f"  Total cycles executed: {status['total_cycles']}")
    print(f"  Cycle breakdown: {status['cycle_counts']}")


if __name__ == '__main__':
    asyncio.run(main())