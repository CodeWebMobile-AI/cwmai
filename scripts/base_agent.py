"""
Base Agent Framework for Collaborative Multi-Agent Systems

Provides the foundation for specialized agents that can analyze, execute,
and collaborate on tasks through a shared blackboard workspace.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import asyncio
import json
import logging
import uuid
from collections import defaultdict

from work_item_types import WorkItem, TaskPriority


class AgentCapability(Enum):
    """Capabilities that agents can possess."""
    ANALYSIS = "analysis"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    SECURITY_ANALYSIS = "security_analysis"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"
    OPTIMIZATION = "optimization"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"


@dataclass
class AgentContext:
    """Context passed to agents for task execution."""
    work_item: WorkItem
    blackboard: Optional['BlackboardWorkspace'] = None
    other_agents: List['BaseAgent'] = field(default_factory=list)
    shared_artifacts: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_artifact(self, key: str, value: Any, agent_id: str) -> None:
        """Add an artifact to shared context."""
        self.shared_artifacts[key] = {
            'value': value,
            'created_by': agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_artifact(self, key: str) -> Optional[Any]:
        """Get an artifact from shared context."""
        artifact = self.shared_artifacts.get(key)
        return artifact['value'] if artifact else None


@dataclass
class AgentResult:
    """Result of agent execution."""
    agent_id: str
    agent_type: str
    success: bool
    output: Any
    artifacts_created: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'success': self.success,
            'output': self.output,
            'artifacts_created': self.artifacts_created,
            'insights': self.insights,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
            'execution_time': self.execution_time,
            'metadata': self.metadata
        }


class BaseAgent(ABC):
    """Abstract base class for all specialized agents."""
    
    def __init__(self, agent_id: Optional[str] = None, ai_brain: Optional[Any] = None):
        """Initialize base agent."""
        self.agent_id = agent_id or f"{self.agent_type}_{uuid.uuid4().hex[:8]}"
        self.ai_brain = ai_brain
        self.capabilities = self._define_capabilities()
        self.performance_metrics = defaultdict(float)
        self.execution_history = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Define the agent type (e.g., 'planner', 'coder', 'tester')."""
        pass
    
    @property
    @abstractmethod
    def persona(self) -> str:
        """Define the agent's persona and approach."""
        pass
    
    @abstractmethod
    def _define_capabilities(self) -> List[AgentCapability]:
        """Define the agent's capabilities."""
        pass
    
    @abstractmethod
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze the task and provide insights."""
        pass
    
    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's primary function."""
        pass
    
    async def collaborate(self, context: AgentContext, other_agent_results: List[AgentResult]) -> AgentResult:
        """Collaborate with other agents based on their results."""
        # Default implementation - agents can override for specific collaboration
        insights = []
        recommendations = []
        
        # Analyze other agents' outputs
        for result in other_agent_results:
            if result.success:
                # Learn from successful agents
                insights.extend([f"Based on {result.agent_type}: {insight}" 
                               for insight in result.insights[:2]])
                
                # Consider their recommendations
                if result.confidence > 0.7:
                    recommendations.extend(result.recommendations[:1])
        
        # Re-execute with collaborative context
        context.execution_history.extend([r.to_dict() for r in other_agent_results])
        
        # Execute with enhanced context
        result = await self.execute(context)
        
        # Add collaborative insights
        result.insights.extend(insights)
        result.recommendations.extend(recommendations)
        
        return result
    
    async def review_artifact(self, artifact_key: str, artifact_value: Any, 
                            created_by: str, context: AgentContext) -> Dict[str, Any]:
        """Review an artifact created by another agent."""
        review = {
            'reviewer': self.agent_id,
            'artifact_key': artifact_key,
            'created_by': created_by,
            'review_type': self.agent_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'feedback': [],
            'approval': True,
            'confidence': 0.0
        }
        
        # Default review - agents should override for specific reviews
        review['feedback'].append(f"Artifact '{artifact_key}' reviewed by {self.agent_type}")
        review['confidence'] = 0.5
        
        return review
    
    def update_performance(self, metric: str, value: float) -> None:
        """Update agent performance metrics."""
        self.performance_metrics[metric] = (
            self.performance_metrics[metric] * 0.9 + value * 0.1
        )
    
    def get_confidence_for_task(self, work_item: WorkItem) -> float:
        """Calculate confidence level for handling a specific task."""
        # Base confidence from task type match
        task_type_confidence = 0.5
        if work_item.task_type in ['code_generation', 'refactoring'] and AgentCapability.CODE_GENERATION in self.capabilities:
            task_type_confidence = 0.8
        elif work_item.task_type in ['testing', 'validation'] and AgentCapability.TESTING in self.capabilities:
            task_type_confidence = 0.8
        elif work_item.task_type in ['security_audit', 'vulnerability_scan'] and AgentCapability.SECURITY_ANALYSIS in self.capabilities:
            task_type_confidence = 0.8
        elif work_item.task_type in ['documentation', 'api_docs'] and AgentCapability.DOCUMENTATION in self.capabilities:
            task_type_confidence = 0.8
        elif work_item.task_type in ['planning', 'architecture'] and AgentCapability.PLANNING in self.capabilities:
            task_type_confidence = 0.8
        
        # Adjust based on performance history
        avg_performance = sum(self.performance_metrics.values()) / max(len(self.performance_metrics), 1)
        
        return min(task_type_confidence * (0.5 + avg_performance * 0.5), 1.0)
    
    async def _call_ai_model(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Call the AI model with agent-specific context."""
        if not self.ai_brain:
            return "No AI brain configured"
        
        # Add agent persona to prompt
        full_prompt = f"{self.persona}\n\n{prompt}"
        
        try:
            response = await self.ai_brain.generate_enhanced_response(full_prompt)
            return response.get('content', '') if response else ""
        except Exception as e:
            self.logger.error(f"Error calling AI model: {e}")
            return f"Error: {str(e)}"
    
    def _parse_ai_response(self, response: str, expected_format: Dict[str, type]) -> Dict[str, Any]:
        """Parse AI response with validation."""
        import re
        
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Validate expected format
                for key, expected_type in expected_format.items():
                    if key not in parsed:
                        if expected_type == list:
                            parsed[key] = []
                        elif expected_type == dict:
                            parsed[key] = {}
                        elif expected_type == str:
                            parsed[key] = ""
                        elif expected_type == float:
                            parsed[key] = 0.0
                        elif expected_type == bool:
                            parsed[key] = False
                
                return parsed
        except Exception as e:
            self.logger.warning(f"Failed to parse AI response: {e}")
        
        # Return default structure
        return {key: ([] if expected_type == list else 
                     {} if expected_type == dict else 
                     "" if expected_type == str else
                     0.0 if expected_type == float else
                     False if expected_type == bool else None)
                for key, expected_type in expected_format.items()}