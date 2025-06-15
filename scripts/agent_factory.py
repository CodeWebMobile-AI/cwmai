"""
Agent Factory for Creating and Managing Specialized Agents

Implements the factory pattern for creating different types of specialized agents
and managing their lifecycle.
"""

from typing import Dict, List, Any, Optional, Type
import logging
from datetime import datetime, timezone

from base_agent import BaseAgent, AgentCapability
from specialized_agents import (
    PlannerAgent,
    CodeAgent,
    TestAgent,
    SecurityAgent,
    DocsAgent
)
from work_item_types import WorkItem, TaskPriority


class AgentFactory:
    """Factory for creating and managing specialized agents."""
    
    # Agent type registry
    AGENT_TYPES: Dict[str, Type[BaseAgent]] = {
        'planner': PlannerAgent,
        'coder': CodeAgent,
        'tester': TestAgent,
        'security': SecurityAgent,
        'documenter': DocsAgent
    }
    
    def __init__(self, ai_brain: Optional[Any] = None):
        """Initialize the agent factory."""
        self.ai_brain = ai_brain
        self.active_agents: Dict[str, BaseAgent] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.logger = logging.getLogger(f"{__name__}.AgentFactory")
        
    def create_agent(self, agent_type: str, agent_id: Optional[str] = None) -> BaseAgent:
        """Create a specialized agent of the specified type."""
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self.AGENT_TYPES[agent_type]
        agent = agent_class(agent_id=agent_id, ai_brain=self.ai_brain)
        
        # Register the agent
        self.active_agents[agent.agent_id] = agent
        
        # Initialize performance tracking
        self.agent_performance[agent.agent_id] = {
            'tasks_completed': 0,
            'success_rate': 1.0,
            'avg_execution_time': 0.0,
            'avg_confidence': 0.0,
            'last_active': datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"Created {agent_type} agent: {agent.agent_id}")
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an active agent by ID."""
        return self.active_agents.get(agent_id)
    
    def create_agent_team(self, task_type: str) -> List[BaseAgent]:
        """Create a team of agents optimized for a specific task type."""
        teams = {
            'feature_implementation': ['planner', 'coder', 'tester', 'documenter'],
            'security_audit': ['security', 'tester', 'documenter'],
            'refactoring': ['coder', 'tester', 'documenter'],
            'documentation': ['documenter', 'planner'],
            'bug_fix': ['coder', 'tester', 'security'],
            'architecture': ['planner', 'coder', 'security', 'documenter'],
            'optimization': ['coder', 'tester', 'security'],
            'testing': ['tester', 'coder', 'documenter']
        }
        
        # Get the appropriate team composition
        team_types = teams.get(task_type, ['planner', 'coder', 'tester', 'security', 'documenter'])
        
        # Create the team
        team = []
        for agent_type in team_types:
            agent = self.create_agent(agent_type)
            team.append(agent)
        
        self.logger.info(f"Created agent team for {task_type}: {[a.agent_id for a in team]}")
        return team
    
    def select_best_agent(self, work_item: WorkItem, available_agents: List[BaseAgent]) -> Optional[BaseAgent]:
        """Select the best agent for a specific work item based on capabilities and performance."""
        if not available_agents:
            return None
        
        best_agent = None
        best_score = 0.0
        
        for agent in available_agents:
            # Calculate suitability score
            confidence = agent.get_confidence_for_task(work_item)
            
            # Get agent's performance history
            perf = self.agent_performance.get(agent.agent_id, {})
            success_rate = perf.get('success_rate', 1.0)
            
            # Adjust for workload (prefer less busy agents)
            workload_factor = 1.0  # Could be enhanced with actual workload tracking
            
            # Calculate overall score
            score = confidence * success_rate * workload_factor
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        if best_agent:
            self.logger.info(f"Selected {best_agent.agent_type} (score: {best_score:.2f}) for task: {work_item.title}")
        
        return best_agent
    
    def update_agent_performance(self, agent_id: str, result: 'AgentResult') -> None:
        """Update agent performance metrics based on execution results."""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                'tasks_completed': 0,
                'success_rate': 1.0,
                'avg_execution_time': 0.0,
                'avg_confidence': 0.0,
                'last_active': datetime.now(timezone.utc).isoformat()
            }
        
        perf = self.agent_performance[agent_id]
        
        # Update metrics
        perf['tasks_completed'] += 1
        perf['last_active'] = datetime.now(timezone.utc).isoformat()
        
        # Update success rate (exponential moving average)
        success_value = 1.0 if result.success else 0.0
        alpha = 0.1  # Learning rate
        perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * success_value
        
        # Update average execution time
        perf['avg_execution_time'] = (
            (perf['avg_execution_time'] * (perf['tasks_completed'] - 1) + result.execution_time) 
            / perf['tasks_completed']
        )
        
        # Update average confidence
        perf['avg_confidence'] = (
            (perf['avg_confidence'] * (perf['tasks_completed'] - 1) + result.confidence) 
            / perf['tasks_completed']
        )
        
        # Update the agent's internal performance metrics
        if agent_id in self.active_agents:
            agent = self.active_agents[agent_id]
            agent.update_performance('success_rate', perf['success_rate'])
            agent.update_performance('avg_confidence', perf['avg_confidence'])
    
    def get_agent_recommendations(self, work_item: WorkItem) -> Dict[str, Any]:
        """Get recommendations for which agents to use for a work item."""
        recommendations = {
            'primary_agents': [],
            'supporting_agents': [],
            'optional_agents': [],
            'rationale': {}
        }
        
        # Determine primary agents based on task type
        if 'code' in work_item.task_type or 'implementation' in work_item.task_type:
            recommendations['primary_agents'] = ['coder', 'tester']
            recommendations['rationale']['coder'] = "Essential for code generation/modification"
            recommendations['rationale']['tester'] = "Required for quality assurance"
        
        if 'security' in work_item.task_type or work_item.priority == TaskPriority.CRITICAL:
            recommendations['primary_agents'].append('security')
            recommendations['rationale']['security'] = "Critical priority requires security review"
        
        if 'planning' in work_item.task_type or 'architecture' in work_item.task_type:
            recommendations['primary_agents'].append('planner')
            recommendations['rationale']['planner'] = "Strategic planning required"
        
        # Supporting agents
        if 'documenter' not in recommendations['primary_agents']:
            recommendations['supporting_agents'].append('documenter')
            recommendations['rationale']['documenter'] = "Documentation should accompany all changes"
        
        # Optional agents based on complexity
        if work_item.estimated_cycles > 5:
            if 'planner' not in recommendations['primary_agents']:
                recommendations['optional_agents'].append('planner')
                recommendations['rationale']['planner'] = "Complex task may benefit from planning"
        
        return recommendations
    
    def get_factory_status(self) -> Dict[str, Any]:
        """Get current status of the agent factory."""
        return {
            'active_agents': len(self.active_agents),
            'agent_types': list(self.active_agents.keys()),
            'performance_summary': {
                agent_id: {
                    'type': agent.agent_type,
                    'tasks_completed': perf['tasks_completed'],
                    'success_rate': f"{perf['success_rate']:.2%}",
                    'avg_confidence': f"{perf['avg_confidence']:.2f}"
                }
                for agent_id, agent in self.active_agents.items()
                if (perf := self.agent_performance.get(agent_id))
            },
            'available_agent_types': list(self.AGENT_TYPES.keys())
        }
    
    def retire_agent(self, agent_id: str) -> bool:
        """Retire an agent from active duty."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
            self.logger.info(f"Retired agent: {agent_id}")
            return True
        return False
    
    def save_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Save agent state for persistence."""
        agent = self.active_agents.get(agent_id)
        if not agent:
            return {}
        
        return {
            'agent_id': agent.agent_id,
            'agent_type': agent.agent_type,
            'performance_metrics': dict(agent.performance_metrics),
            'execution_history': agent.execution_history[-10:],  # Last 10 executions
            'factory_performance': self.agent_performance.get(agent_id, {})
        }
    
    def restore_agent_state(self, state: Dict[str, Any]) -> Optional[BaseAgent]:
        """Restore agent from saved state."""
        agent_type = state.get('agent_type')
        agent_id = state.get('agent_id')
        
        if not agent_type or agent_type not in self.AGENT_TYPES:
            return None
        
        # Create agent with saved ID
        agent = self.create_agent(agent_type, agent_id)
        
        # Restore performance metrics
        if 'performance_metrics' in state:
            for metric, value in state['performance_metrics'].items():
                agent.performance_metrics[metric] = value
        
        # Restore execution history
        if 'execution_history' in state:
            agent.execution_history = state['execution_history']
        
        # Restore factory performance tracking
        if 'factory_performance' in state:
            self.agent_performance[agent_id] = state['factory_performance']
        
        return agent