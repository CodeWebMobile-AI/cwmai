"""
Enhanced Swarm Intelligence

Enhanced version of the existing swarm intelligence system with integrated
logging, metrics, error analysis, and performance optimization.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Import the original swarm intelligence
from scripts.dynamic_swarm import DynamicSwarmIntelligence, DynamicSwarmAgent
from scripts.swarm_intelligence import RealSwarmIntelligence, RealSwarmAgent, AgentRole

# Import the new intelligence components
from scripts.worker_intelligence_integration import (
    IntelligentWorkerMixin,
    WorkerEnhancementConfig,
    WorkerIntelligenceCoordinator,
    enhance_worker_method
)
from scripts.worker_logging_config import setup_worker_logger, WorkerOperationContext
from scripts.worker_intelligence_hub import WorkerSpecialization


class EnhancedSwarmAgent(DynamicSwarmAgent, IntelligentWorkerMixin):
    """Enhanced swarm agent with intelligence capabilities."""
    
    def __init__(self, id: str, role: AgentRole, model_name: str, 
                 expertise_areas: List[str], persona: str, ai_brain, 
                 task_history: List[Dict[str, Any]] = None,
                 system_context: Dict[str, Any] = None):
        """Initialize enhanced swarm agent."""
        # Initialize base agent
        DynamicSwarmAgent.__init__(
            self, id, role, model_name, expertise_areas, persona, ai_brain,
            task_history or [], system_context or {}
        )
        
        # Initialize intelligence capabilities
        config = WorkerEnhancementConfig(
            worker_specialization=self._role_to_specialization(role),
            enable_logging=True,
            enable_metrics=True,
            enable_error_analysis=True
        )
        IntelligentWorkerMixin.__init__(self, id, config)
    
    def _role_to_specialization(self, role: AgentRole) -> WorkerSpecialization:
        """Convert agent role to worker specialization."""
        role_mapping = {
            AgentRole.ARCHITECT: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.DEVELOPER: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.TESTER: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.SECURITY: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.PERFORMANCE: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.RESEARCHER: WorkerSpecialization.RESEARCH_ANALYSIS,
            AgentRole.STRATEGIST: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.LEARNER: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.REVIEWER: WorkerSpecialization.SWARM_INTELLIGENCE,
            AgentRole.ORCHESTRATOR: WorkerSpecialization.TASK_COORDINATION
        }
        return role_mapping.get(role, WorkerSpecialization.SWARM_INTELLIGENCE)
    
    @enhance_worker_method("task_analysis")
    async def analyze_task(self, task: Dict[str, Any], 
                          other_insights: List[Dict[str, Any]] = None,
                          iteration: int = 1) -> Dict[str, Any]:
        """Enhanced task analysis with intelligence tracking."""
        # Update progress tracking
        if self.work_item_tracker:
            # This would be set by the swarm coordination
            pass
        
        # Call original analysis method
        return await super().analyze_task(task, other_insights, iteration)
    
    def _parse_ai_response(self, response):
        """Enhanced response parsing with better error handling."""
        try:
            # Use parent's parsing with enhanced logging
            with WorkerOperationContext(self.id, "response_parsing"):
                result = super()._parse_ai_response(response)
                
                # Log parsing success
                if self.logger:
                    self.logger.debug(f"Successfully parsed response with {len(result.get('key_points', []))} key points")
                
                return result
                
        except Exception as e:
            # Enhanced error handling
            if self.error_analyzer:
                self.error_analyzer.register_error(
                    self.id, "response_parsing", e, {
                        'response_length': len(str(response)),
                        'agent_role': self.role.value,
                        'model': self.model_name
                    }
                )
            
            # Return safe fallback
            return {
                'key_points': [],
                'challenges': [],
                'recommendations': [],
                'priority': 5,
                'complexity': 'unknown',
                'confidence': 0,
                'alignment_score': 0,
                'parse_error': str(e)
            }


class EnhancedSwarmIntelligence(DynamicSwarmIntelligence, IntelligentWorkerMixin):
    """Enhanced swarm intelligence with full intelligence integration."""
    
    def __init__(self, ai_brain, learning_system=None, charter_system=None,
                 intelligence_coordinator: WorkerIntelligenceCoordinator = None):
        """Initialize enhanced swarm intelligence."""
        # Initialize base swarm
        DynamicSwarmIntelligence.__init__(self, ai_brain, learning_system, charter_system)
        
        # Initialize intelligence capabilities
        config = WorkerEnhancementConfig(
            worker_specialization=WorkerSpecialization.SWARM_INTELLIGENCE,
            enable_logging=True,
            enable_intelligence=True,
            enable_metrics=True,
            enable_error_analysis=True,
            enable_work_tracking=True,
            enable_status_reporting=True
        )
        IntelligentWorkerMixin.__init__(self, "swarm_coordinator", config)
        
        # Set intelligence coordinator
        self.intelligence_coordinator = intelligence_coordinator
        if intelligence_coordinator:
            self.set_intelligence_components(
                intelligence_coordinator.intelligence_hub,
                intelligence_coordinator.metrics_collector,
                intelligence_coordinator.error_analyzer,
                intelligence_coordinator.work_item_tracker,
                intelligence_coordinator.status_reporter
            )
        
        # Convert agents to enhanced agents
        self._convert_to_enhanced_agents()
    
    def _convert_to_enhanced_agents(self):
        """Convert base agents to enhanced agents."""
        enhanced_agents = []
        
        for agent in self.agents:
            enhanced_agent = EnhancedSwarmAgent(
                id=agent.id,
                role=agent.role,
                model_name=agent.model_name,
                expertise_areas=agent.expertise_areas,
                persona=agent.persona,
                ai_brain=agent.ai_brain,
                task_history=agent.task_history,
                system_context=getattr(agent, 'system_context', {})
            )
            
            # Set intelligence components for agent
            if self.intelligence_coordinator:
                enhanced_agent.set_intelligence_components(
                    self.intelligence_coordinator.intelligence_hub,
                    self.intelligence_coordinator.metrics_collector,
                    self.intelligence_coordinator.error_analyzer,
                    self.intelligence_coordinator.work_item_tracker,
                    self.intelligence_coordinator.status_reporter
                )
            
            enhanced_agents.append(enhanced_agent)
        
        self.agents = enhanced_agents
        self.logger.info(f"Converted {len(enhanced_agents)} agents to enhanced agents")
    
    @enhance_worker_method("swarm_analysis")
    async def process_task_swarm(self, task: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced swarm task processing with intelligence tracking."""
        # Create work item for tracking
        work_item_id = None
        if self.work_item_tracker:
            work_item_id = self.work_item_tracker.create_work_item(
                title=f"Swarm Analysis: {task.get('title', 'Unknown')}",
                description=f"Swarm intelligence analysis for {task.get('type', 'unknown')} task",
                work_type="swarm_analysis",
                created_by="swarm_coordinator",
                context=task
            )
            self.work_item_tracker.assign_to_worker(work_item_id, "swarm_coordinator")
        
        try:
            # Update progress
            if work_item_id:
                self.work_item_tracker.update_progress(work_item_id, 10, "Starting swarm analysis")
            
            # Call original swarm processing
            result = await super().process_task_swarm(task, context)
            
            # Update progress throughout the process
            if work_item_id:
                self.work_item_tracker.update_progress(work_item_id, 50, "Individual analysis complete")
                self.work_item_tracker.update_progress(work_item_id, 75, "Cross-pollination complete")
                self.work_item_tracker.update_progress(work_item_id, 90, "Consensus building complete")
                self.work_item_tracker.update_progress(work_item_id, 100, "Swarm analysis complete")
            
            # Complete work item
            if work_item_id:
                self.work_item_tracker.complete_work(work_item_id, "swarm_coordinator", {
                    'consensus_priority': result.get('consensus', {}).get('consensus_priority', 5),
                    'agents_used': len(result.get('individual_analyses', [])),
                    'duration': result.get('duration_seconds', 0)
                })
            
            # Enhanced result with intelligence metrics
            if self.intelligence_hub:
                system_status = self.intelligence_hub.get_system_status()
                result['intelligence_metrics'] = {
                    'system_health': system_status.get('system_metrics', {}),
                    'swarm_performance': self.swarm_performance_metrics
                }
            
            return result
            
        except Exception as e:
            # Handle failure
            if work_item_id:
                self.work_item_tracker.fail_work(work_item_id, "swarm_coordinator", str(e))
            
            # Log enhanced error information
            self.logger.error(f"Swarm analysis failed for task {task.get('id', 'unknown')}: {e}")
            raise
    
    async def _phase_individual_analysis(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced individual analysis with intelligence tracking."""
        self.logger.info(f"[ENHANCED_SWARM] Starting individual analysis with {len(self.agents)} agents")
        
        analyses = []
        
        # Track each agent's analysis
        for i, agent in enumerate(self.agents):
            agent_task_id = f"individual_analysis_{task.get('id', 'unknown')}_{agent.id}"
            
            try:
                # Use agent's intelligent task execution
                async with agent.intelligent_task_execution(
                    agent_task_id, "individual_analysis", {
                        'task_type': task.get('type', 'unknown'),
                        'agent_role': agent.role.value,
                        'iteration': 1
                    }
                ):
                    analysis = await agent.analyze_task(task, iteration=1)
                    analyses.append(analysis)
                    
                    # Update swarm metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_metric(
                            "swarm.agent_analysis_success", 1,
                            labels={"agent_id": agent.id, "agent_role": agent.role.value}
                        )
                
            except Exception as e:
                self.logger.error(f"Agent {agent.id} individual analysis failed: {e}")
                
                # Create fallback analysis
                analyses.append({
                    'agent_id': agent.id,
                    'agent_role': agent.role.value,
                    'error': str(e),
                    'key_points': [],
                    'challenges': [],
                    'recommendations': [],
                    'priority': 5,
                    'complexity': 'unknown',
                    'confidence': 0,
                    'alignment_score': 0
                })
                
                # Record failure metric
                if self.metrics_collector:
                    self.metrics_collector.record_metric(
                        "swarm.agent_analysis_failure", 1,
                        labels={"agent_id": agent.id, "agent_role": agent.role.value}
                    )
        
        self.logger.info(f"[ENHANCED_SWARM] Individual analysis completed: {len(analyses)} analyses")
        return analyses
    
    async def _phase_enhanced_cross_pollination(self, task: Dict[str, Any],
                                               initial_analyses: List[Dict[str, Any]],
                                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced cross-pollination with intelligence tracking."""
        self.logger.info(f"[ENHANCED_SWARM] Starting cross-pollination with {len(initial_analyses)} initial analyses")
        
        refined_analyses = []
        
        for i, agent in enumerate(self.agents):
            agent_task_id = f"cross_pollination_{task.get('id', 'unknown')}_{agent.id}"
            
            try:
                # Get other agents' analyses
                other_analyses = [a for j, a in enumerate(initial_analyses) if j != i]
                
                # Use agent's intelligent task execution
                async with agent.intelligent_task_execution(
                    agent_task_id, "cross_pollination", {
                        'task_type': task.get('type', 'unknown'),
                        'agent_role': agent.role.value,
                        'other_analyses_count': len(other_analyses),
                        'iteration': 2
                    }
                ):
                    refined_analysis = await agent.analyze_task(task, other_analyses, iteration=2)
                    refined_analyses.append(refined_analysis)
                    
                    # Record success metric
                    if self.metrics_collector:
                        self.metrics_collector.record_metric(
                            "swarm.cross_pollination_success", 1,
                            labels={"agent_id": agent.id, "agent_role": agent.role.value}
                        )
                
            except Exception as e:
                self.logger.error(f"Agent {agent.id} cross-pollination failed: {e}")
                
                # Fall back to initial analysis
                if i < len(initial_analyses):
                    refined_analyses.append(initial_analyses[i])
                else:
                    refined_analyses.append({
                        'agent_id': agent.id,
                        'agent_role': agent.role.value,
                        'error': str(e),
                        'key_points': [],
                        'challenges': [],
                        'recommendations': [],
                        'priority': 5,
                        'complexity': 'unknown',
                        'confidence': 0,
                        'alignment_score': 0
                    })
        
        self.logger.info(f"[ENHANCED_SWARM] Cross-pollination completed: {len(refined_analyses)} refined analyses")
        return refined_analyses
    
    def get_enhanced_analytics(self) -> Dict[str, Any]:
        """Get enhanced analytics including intelligence metrics."""
        base_analytics = self.get_swarm_analytics()
        
        enhanced_analytics = {
            **base_analytics,
            'intelligence_integration': {
                'components_enabled': {
                    'logging': self.config.enable_logging,
                    'intelligence_hub': self.config.enable_intelligence,
                    'metrics': self.config.enable_metrics,
                    'error_analysis': self.config.enable_error_analysis,
                    'work_tracking': self.config.enable_work_tracking,
                    'status_reporting': self.config.enable_status_reporting
                },
                'coordinator_available': self.intelligence_coordinator is not None
            }
        }
        
        # Add intelligence hub metrics if available
        if self.intelligence_hub:
            system_status = self.intelligence_hub.get_system_status()
            enhanced_analytics['intelligence_hub_status'] = system_status
        
        # Add metrics collector data if available
        if self.metrics_collector:
            dashboard = self.metrics_collector.get_system_performance_dashboard()
            enhanced_analytics['performance_dashboard'] = dashboard
        
        # Add error analysis if available
        if self.error_analyzer:
            error_summary = self.error_analyzer.get_error_summary(hours=24)
            enhanced_analytics['error_analysis'] = error_summary
        
        return enhanced_analytics


# Factory function to create enhanced swarm
def create_enhanced_swarm(ai_brain, learning_system=None, charter_system=None,
                         enable_coordinator: bool = True) -> tuple[EnhancedSwarmIntelligence, Optional[WorkerIntelligenceCoordinator]]:
    """Create enhanced swarm intelligence with optional coordinator.
    
    Args:
        ai_brain: AI brain for agents
        learning_system: Optional learning system
        charter_system: Optional charter system
        enable_coordinator: Whether to create intelligence coordinator
        
    Returns:
        Tuple of (enhanced_swarm, coordinator)
    """
    coordinator = None
    
    if enable_coordinator:
        coordinator = WorkerIntelligenceCoordinator()
    
    enhanced_swarm = EnhancedSwarmIntelligence(
        ai_brain, learning_system, charter_system, coordinator
    )
    
    return enhanced_swarm, coordinator


# Example usage
async def demonstrate_enhanced_swarm():
    """Demonstrate the enhanced swarm intelligence capabilities."""
    from ai_brain import IntelligentAIBrain  # Assuming this exists
    
    # Create AI brain (mock for demonstration)
    class MockAIBrain:
        async def generate_enhanced_response(self, prompt, model=None):
            return {
                'content': '{"key_points": ["Test insight"], "challenges": ["Test challenge"], "recommendations": ["Test recommendation"], "priority": 7, "complexity": "medium", "confidence": 0.8, "alignment_score": 0.9}'
            }
    
    ai_brain = MockAIBrain()
    
    # Create enhanced swarm
    enhanced_swarm, coordinator = create_enhanced_swarm(ai_brain, enable_coordinator=True)
    
    if coordinator:
        await coordinator.start()
    
    try:
        # Create test task
        test_task = {
            'id': 'test_task_1',
            'type': 'feature_analysis',
            'title': 'Implement user authentication',
            'description': 'Add JWT-based authentication to the application',
            'requirements': ['Security', 'Scalability', 'User experience']
        }
        
        # Process task with enhanced swarm
        result = await enhanced_swarm.process_task_swarm(test_task)
        
        print("Enhanced Swarm Analysis Result:")
        print(f"- Task ID: {result.get('task_id')}")
        print(f"- Duration: {result.get('duration_seconds', 0):.2f} seconds")
        print(f"- Agents Used: {len(result.get('individual_analyses', []))}")
        print(f"- Consensus Priority: {result.get('consensus', {}).get('consensus_priority', 'N/A')}")
        
        if 'intelligence_metrics' in result:
            print(f"- Intelligence Integration: ✓")
        
        # Get enhanced analytics
        analytics = enhanced_swarm.get_enhanced_analytics()
        print(f"\nSystem Analytics:")
        print(f"- Total Analyses: {analytics.get('total_analyses', 0)}")
        print(f"- Intelligence Components: {analytics.get('intelligence_integration', {}).get('components_enabled', {})}")
        
        # Get dashboard data from coordinator
        if coordinator:
            dashboard = coordinator.get_system_dashboard()
            print(f"- System Health: {dashboard.get('system_health', {}).get('status', 'unknown')}")
            print(f"- Active Workers: {dashboard.get('system_health', {}).get('total_workers', 0)}")
    
    finally:
        if coordinator:
            await coordinator.stop()


if __name__ == "__main__":
    asyncio.run(demonstrate_enhanced_swarm())