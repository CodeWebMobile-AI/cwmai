"""
Multi-Agent System for Complex Task Handling

Provides specialized agents that work together to handle complex queries
and tasks more effectively than a single AI.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from scripts.http_ai_client import HTTPAIClient
from scripts.tool_calling_system import ToolCallingSystem
from scripts.semantic_memory_system import SemanticMemorySystem
from scripts.state_manager import StateManager
from scripts.task_manager import TaskManager


class AgentRole(Enum):
    """Roles that agents can play."""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    DEBUGGER = "debugger"
    CREATOR = "creator"
    MONITOR = "monitor"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    content: Any
    message_type: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class AgentCapability:
    """Represents a capability an agent has."""
    name: str
    description: str
    required_context: List[str]
    confidence: float = 1.0


class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, name: str, role: AgentRole, description: str):
        """Initialize base agent.
        
        Args:
            name: Agent's unique name
            role: Agent's role
            description: What the agent does
        """
        self.name = name
        self.role = role
        self.description = description
        self.capabilities: List[AgentCapability] = []
        self.logger = logging.getLogger(f"Agent.{name}")
        self.ai_client = HTTPAIClient(enable_round_robin=True)
        self.message_history: List[AgentMessage] = []
        self.state = {}
        
    @abstractmethod
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task with given context.
        
        Args:
            task: Task to process
            context: Shared context from coordinator
            
        Returns:
            Processing result
        """
        pass
    
    async def can_handle(self, task: Dict[str, Any]) -> float:
        """Determine if this agent can handle the task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Default implementation using capability matching
        task_type = task.get('type', '')
        task_description = task.get('description', '')
        
        max_confidence = 0.0
        for capability in self.capabilities:
            if capability.name in task_type or capability.name in task_description.lower():
                max_confidence = max(max_confidence, capability.confidence)
        
        return max_confidence
    
    async def collaborate(self, other_agent: 'BaseAgent', message: AgentMessage) -> Optional[AgentMessage]:
        """Collaborate with another agent.
        
        Args:
            other_agent: Agent to collaborate with
            message: Message to send
            
        Returns:
            Response message if any
        """
        self.logger.info(f"Collaborating with {other_agent.name}: {message.message_type}")
        
        # Process the collaboration
        response = await other_agent.receive_message(message)
        
        if response:
            self.message_history.append(response)
            
        return response
    
    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and process a message from another agent.
        
        Args:
            message: Incoming message
            
        Returns:
            Response message if needed
        """
        self.message_history.append(message)
        
        # Default: acknowledge receipt
        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            content={"acknowledged": True},
            message_type="ack"
        )


class CoordinatorAgent(BaseAgent):
    """Coordinates work between multiple agents."""
    
    def __init__(self):
        super().__init__(
            name="Coordinator",
            role=AgentRole.COORDINATOR,
            description="Coordinates work between specialized agents"
        )
        
        self.capabilities = [
            AgentCapability("task_distribution", "Distribute tasks to appropriate agents", []),
            AgentCapability("result_synthesis", "Combine results from multiple agents", []),
            AgentCapability("conflict_resolution", "Resolve conflicts between agent outputs", [])
        ]
        
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate task execution across agents."""
        self.logger.info(f"Coordinating task: {task.get('description', 'Unknown')}")
        
        # Analyze the task to determine required agents
        required_agents = await self._determine_required_agents(task)
        
        # Create execution plan
        plan = await self._create_execution_plan(task, required_agents)
        
        # Return coordination result
        return {
            "coordinator": self.name,
            "plan": plan,
            "required_agents": required_agents,
            "estimated_complexity": len(required_agents)
        }
    
    async def _determine_required_agents(self, task: Dict[str, Any]) -> List[str]:
        """Determine which agents are needed for a task."""
        prompt = f"""Analyze this task and determine which specialized agents are needed:

Task: {json.dumps(task, indent=2)}

Available agent types:
- Executor: Executes commands and actions
- Analyzer: Analyzes data and provides insights  
- Planner: Creates detailed execution plans
- Researcher: Researches information and gathers context
- Debugger: Debugs issues and errors
- Creator: Creates new content or code
- Monitor: Monitors system state and health

Return a JSON list of required agent types."""
        
        response = await self.ai_client.generate_enhanced_response(prompt, prefill='[')
        
        try:
            content = response.get('content', '[]')
            if not content.startswith('['):
                content = '[' + content
            return json.loads(content)
        except:
            # Fallback based on task type
            task_type = task.get('type', '').lower()
            if 'create' in task_type:
                return ['Creator', 'Executor']
            elif 'analyze' in task_type:
                return ['Analyzer', 'Researcher']
            elif 'debug' in task_type or 'fix' in task_type:
                return ['Debugger', 'Analyzer']
            else:
                return ['Executor']
    
    async def _create_execution_plan(self, task: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """Create a detailed execution plan."""
        return {
            "phases": [
                {"phase": "analysis", "agents": ["Analyzer"], "duration": "2s"},
                {"phase": "planning", "agents": ["Planner"], "duration": "3s"},
                {"phase": "execution", "agents": agents, "duration": "5s"},
                {"phase": "validation", "agents": ["Monitor"], "duration": "2s"}
            ],
            "parallel_execution": len(agents) > 2,
            "estimated_time": f"{2 + 3 + 5 + 2}s"
        }


class ExecutorAgent(BaseAgent):
    """Executes commands and actions."""
    
    def __init__(self, tool_system: ToolCallingSystem):
        super().__init__(
            name="Executor",
            role=AgentRole.EXECUTOR,
            description="Executes commands and actions using available tools"
        )
        
        self.tool_system = tool_system
        
        self.capabilities = [
            AgentCapability("command_execution", "Execute system commands", ["command"]),
            AgentCapability("tool_calling", "Call registered tools", ["tool_name", "parameters"]),
            AgentCapability("api_interaction", "Interact with APIs", ["api_endpoint"])
        ]
        
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the task."""
        self.logger.info(f"Executing task: {task}")
        
        task_type = task.get('type', '')
        
        if task_type == 'tool_call':
            # Execute tool call
            tool_name = task.get('tool_name')
            params = task.get('parameters', {})
            
            if tool_name:
                result = await self.tool_system.call_tool(tool_name, **params)
                return {
                    "agent": self.name,
                    "action": "tool_execution",
                    "tool": tool_name,
                    "result": result
                }
        
        elif task_type == 'command':
            # Execute command
            command = task.get('command')
            if command:
                result = await self.tool_system.call_tool('execute_command', command=command)
                return {
                    "agent": self.name,
                    "action": "command_execution",
                    "command": command,
                    "result": result
                }
        
        # General execution using AI
        prompt = f"""Execute this task:
{json.dumps(task, indent=2)}

Context:
{json.dumps(context, indent=2)}

Determine what tools to call and execute them."""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        # Parse and execute any tool calls
        tool_results = await self.tool_system.parse_and_execute_tool_calls(response.get('content', ''))
        
        return {
            "agent": self.name,
            "action": "ai_guided_execution",
            "ai_response": response.get('content'),
            "tool_results": tool_results
        }


class AnalyzerAgent(BaseAgent):
    """Analyzes data and provides insights."""
    
    def __init__(self, memory_system: SemanticMemorySystem):
        super().__init__(
            name="Analyzer",
            role=AgentRole.ANALYZER,
            description="Analyzes data, patterns, and provides insights"
        )
        
        self.memory_system = memory_system
        
        self.capabilities = [
            AgentCapability("data_analysis", "Analyze structured data", ["data"]),
            AgentCapability("pattern_recognition", "Identify patterns", ["dataset"]),
            AgentCapability("anomaly_detection", "Detect anomalies", ["metrics"]),
            AgentCapability("performance_analysis", "Analyze performance", ["metrics", "baseline"])
        ]
        
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the task or data."""
        self.logger.info(f"Analyzing: {task.get('description', 'Unknown')}")
        
        # Get relevant memories for context
        relevant_memories = await self.memory_system.search_memories(
            task.get('description', ''),
            k=5
        )
        
        # Build analysis prompt
        memory_context = "\n".join([
            f"- {m.content} (relevance: {m.relevance_score:.2f})"
            for m in relevant_memories
        ])
        
        prompt = f"""Analyze this task/data:

Task: {json.dumps(task, indent=2)}

Context: {json.dumps(context, indent=2)}

Relevant past insights:
{memory_context}

Provide:
1. Key findings
2. Patterns identified
3. Recommendations
4. Potential issues"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        # Store the analysis as a memory
        analysis_content = response.get('content', '')
        await self.memory_system.add_memory(
            content=f"Analysis: {task.get('description', 'Unknown')}\n{analysis_content}",
            metadata={
                "type": "analysis",
                "agent": self.name,
                "task_id": task.get('id')
            }
        )
        
        return {
            "agent": self.name,
            "analysis": analysis_content,
            "memories_used": len(relevant_memories),
            "confidence": 0.85
        }


class PlannerAgent(BaseAgent):
    """Creates detailed execution plans."""
    
    def __init__(self):
        super().__init__(
            name="Planner",
            role=AgentRole.PLANNER,
            description="Creates detailed plans for complex tasks"
        )
        
        self.capabilities = [
            AgentCapability("task_decomposition", "Break down complex tasks", ["task"]),
            AgentCapability("dependency_analysis", "Analyze task dependencies", ["tasks"]),
            AgentCapability("resource_planning", "Plan resource allocation", ["requirements"]),
            AgentCapability("timeline_creation", "Create execution timelines", ["tasks", "constraints"])
        ]
        
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed plan for the task."""
        prompt = f"""Create a detailed execution plan for this task:

Task: {json.dumps(task, indent=2)}
Context: {json.dumps(context, indent=2)}

Provide a step-by-step plan with:
1. Clear phases
2. Dependencies between steps
3. Required resources/tools
4. Success criteria
5. Potential risks and mitigations"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        return {
            "agent": self.name,
            "plan": response.get('content'),
            "estimated_steps": task.get('complexity', 5),
            "planning_complete": True
        }


class ResearcherAgent(BaseAgent):
    """Researches information and gathers context."""
    
    def __init__(self, memory_system: SemanticMemorySystem):
        super().__init__(
            name="Researcher",
            role=AgentRole.RESEARCHER,
            description="Researches information and gathers relevant context"
        )
        
        self.memory_system = memory_system
        
        self.capabilities = [
            AgentCapability("information_gathering", "Gather relevant information", ["topic"]),
            AgentCapability("context_building", "Build comprehensive context", ["query"]),
            AgentCapability("fact_checking", "Verify information", ["claims"]),
            AgentCapability("trend_analysis", "Analyze trends", ["data_points"])
        ]
        
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Research the topic."""
        topic = task.get('topic', task.get('description', ''))
        
        # Search memories
        memories = await self.memory_system.search_memories(topic, k=10)
        
        # Get summary if many memories
        summary = ""
        if len(memories) > 5:
            summary = await self.memory_system.summarize_memory_cluster(topic)
        
        # Research using AI
        prompt = f"""Research this topic thoroughly:

Topic: {topic}

Known information:
{summary if summary else 'No prior knowledge found'}

Provide:
1. Overview
2. Key facts
3. Recent developments
4. Best practices
5. Common challenges
6. Recommendations"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        # Store research as memory
        research_content = response.get('content', '')
        await self.memory_system.add_memory(
            content=f"Research on {topic}:\n{research_content}",
            metadata={
                "type": "research",
                "agent": self.name,
                "topic": topic
            }
        )
        
        return {
            "agent": self.name,
            "research": research_content,
            "memories_found": len(memories),
            "prior_knowledge": summary or "None"
        }


class DebuggerAgent(BaseAgent):
    """Debugs issues and errors."""
    
    def __init__(self):
        super().__init__(
            name="Debugger",
            role=AgentRole.DEBUGGER,
            description="Debugs issues, errors, and system problems"
        )
        
        self.capabilities = [
            AgentCapability("error_analysis", "Analyze errors", ["error", "context"]),
            AgentCapability("root_cause_analysis", "Find root causes", ["symptoms"]),
            AgentCapability("fix_suggestion", "Suggest fixes", ["issue"]),
            AgentCapability("test_generation", "Generate tests", ["scenario"])
        ]
        
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Debug the issue."""
        issue = task.get('issue', task.get('description', ''))
        error = task.get('error')
        
        prompt = f"""Debug this issue:

Issue: {issue}
Error: {error if error else 'No specific error provided'}
Context: {json.dumps(context, indent=2)}

Provide:
1. Root cause analysis
2. Step-by-step debugging approach
3. Potential fixes
4. Preventive measures
5. Test cases to verify the fix"""
        
        response = await self.ai_client.generate_enhanced_response(prompt)
        
        return {
            "agent": self.name,
            "debugging_analysis": response.get('content'),
            "severity": self._assess_severity(issue, error),
            "debugging_complete": True
        }
    
    def _assess_severity(self, issue: str, error: Any) -> str:
        """Assess issue severity."""
        if error and ('critical' in str(error).lower() or 'fatal' in str(error).lower()):
            return "critical"
        elif 'not working' in issue.lower() or 'broken' in issue.lower():
            return "high"
        elif 'slow' in issue.lower() or 'performance' in issue.lower():
            return "medium"
        else:
            return "low"


class MultiAgentSystem:
    """Orchestrates multiple agents to handle complex tasks."""
    
    def __init__(self, tool_system: ToolCallingSystem, memory_system: SemanticMemorySystem):
        """Initialize the multi-agent system."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {
            "Coordinator": CoordinatorAgent(),
            "Executor": ExecutorAgent(tool_system),
            "Analyzer": AnalyzerAgent(memory_system),
            "Planner": PlannerAgent(),
            "Researcher": ResearcherAgent(memory_system),
            "Debugger": DebuggerAgent()
        }
        
        # Shared context
        self.shared_context = {
            "session_start": datetime.now(timezone.utc).isoformat(),
            "completed_tasks": [],
            "active_agents": []
        }
        
    async def process_request(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user request using multiple agents.
        
        Args:
            user_input: The user's request
            context: Additional context
            
        Returns:
            Processing result with agent contributions
        """
        # Create task from user input
        task = {
            "id": f"task_{datetime.now(timezone.utc).timestamp()}",
            "description": user_input,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context or {}
        }
        
        # Merge contexts
        full_context = {**self.shared_context, **(context or {})}
        
        # Start with coordinator
        coordinator = self.agents["Coordinator"]
        coordination_result = await coordinator.process(task, full_context)
        
        # Get required agents
        required_agents = coordination_result.get('required_agents', ['Executor'])
        
        # Execute with required agents
        agent_results = {}
        
        # Process in parallel where possible
        if coordination_result.get('plan', {}).get('parallel_execution'):
            # Parallel execution
            agent_tasks = []
            for agent_name in required_agents:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    agent_tasks.append(self._execute_agent(agent, task, full_context))
            
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            for i, agent_name in enumerate(required_agents):
                if not isinstance(results[i], Exception):
                    agent_results[agent_name] = results[i]
                else:
                    agent_results[agent_name] = {"error": str(results[i])}
        else:
            # Sequential execution
            for agent_name in required_agents:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    try:
                        result = await self._execute_agent(agent, task, full_context)
                        agent_results[agent_name] = result
                        
                        # Update context with results
                        full_context[f"{agent_name}_result"] = result
                    except Exception as e:
                        self.logger.error(f"Agent {agent_name} error: {e}")
                        agent_results[agent_name] = {"error": str(e)}
        
        # Synthesize results
        final_result = await self._synthesize_results(task, agent_results, coordination_result)
        
        # Update shared context
        self.shared_context['completed_tasks'].append(task['id'])
        
        return final_result
    
    async def _execute_agent(self, agent: BaseAgent, task: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent."""
        self.logger.info(f"Executing agent: {agent.name}")
        
        # Update active agents
        self.shared_context['active_agents'].append(agent.name)
        
        try:
            result = await agent.process(task, context)
            return result
        finally:
            # Remove from active agents
            self.shared_context['active_agents'].remove(agent.name)
    
    async def _synthesize_results(self, task: Dict[str, Any], agent_results: Dict[str, Any],
                                coordination_result: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple agents."""
        # Count successful agents
        successful_agents = sum(1 for r in agent_results.values() if 'error' not in r)
        
        # Build summary
        summary_parts = []
        
        for agent_name, result in agent_results.items():
            if 'error' not in result:
                if agent_name == "Analyzer" and 'analysis' in result:
                    summary_parts.append(f"Analysis: {result['analysis'][:200]}...")
                elif agent_name == "Executor" and 'result' in result:
                    summary_parts.append(f"Execution: {result['result']}")
                elif agent_name == "Researcher" and 'research' in result:
                    summary_parts.append(f"Research: {result['research'][:200]}...")
                elif agent_name == "Planner" and 'plan' in result:
                    summary_parts.append(f"Plan created with {result.get('estimated_steps', 'unknown')} steps")
                elif agent_name == "Debugger" and 'debugging_analysis' in result:
                    summary_parts.append(f"Debug: {result['severity']} severity issue analyzed")
        
        return {
            "task_id": task['id'],
            "user_input": task['description'],
            "coordination": coordination_result,
            "agent_results": agent_results,
            "summary": "\n\n".join(summary_parts),
            "successful_agents": successful_agents,
            "total_agents_used": len(agent_results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_agent_recommendations(self, user_input: str) -> List[Dict[str, Any]]:
        """Get recommendations for which agents to use.
        
        Args:
            user_input: The user's input
            
        Returns:
            List of agent recommendations with confidence scores
        """
        recommendations = []
        
        task = {"description": user_input, "type": "analysis"}
        
        for agent_name, agent in self.agents.items():
            if agent_name != "Coordinator":  # Coordinator is always involved
                confidence = await agent.can_handle(task)
                if confidence > 0.0:
                    recommendations.append({
                        "agent": agent_name,
                        "role": agent.role.value,
                        "confidence": confidence,
                        "description": agent.description
                    })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations