"""
PlannerAgent - Strategic Planning and Project Management

Specializes in high-level strategy, project decomposition, and roadmap creation.
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

import sys
sys.path.append('..')

from base_agent import BaseAgent, AgentCapability, AgentContext, AgentResult


class PlannerAgent(BaseAgent):
    """Agent specialized in strategic planning and project management."""
    
    @property
    def agent_type(self) -> str:
        return "planner"
    
    @property
    def persona(self) -> str:
        return """You are a senior strategic planner and project architect with 20+ years of experience 
        in software development. You excel at breaking down complex projects into actionable phases, 
        identifying dependencies, and creating realistic timelines. You think in terms of milestones, 
        deliverables, and risk mitigation. You always consider resource allocation, team capabilities, 
        and potential blockers."""
    
    def _define_capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability.PLANNING,
            AgentCapability.ANALYSIS,
            AgentCapability.ARCHITECTURE
        ]
    
    async def analyze(self, context: AgentContext) -> Dict[str, Any]:
        """Analyze the task from a planning perspective."""
        work_item = context.work_item
        
        prompt = f"""
        Analyze this task from a strategic planning perspective:
        
        Task: {work_item.title}
        Description: {work_item.description}
        Type: {work_item.task_type}
        Priority: {work_item.priority.name}
        
        Provide analysis including:
        1. Project scope and complexity assessment
        2. Key phases or milestones needed
        3. Critical dependencies and prerequisites
        4. Resource requirements (time, skills, tools)
        5. Potential risks and mitigation strategies
        6. Success criteria and metrics
        
        Format as JSON with keys: scope_assessment, phases, dependencies, 
        resources, risks, success_criteria, estimated_duration, complexity_score
        """
        
        response = await self._call_ai_model(prompt)
        
        expected_format = {
            'scope_assessment': str,
            'phases': list,
            'dependencies': list,
            'resources': dict,
            'risks': list,
            'success_criteria': list,
            'estimated_duration': str,
            'complexity_score': float
        }
        
        return self._parse_ai_response(response, expected_format)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Create a strategic plan for the task."""
        start_time = time.time()
        
        try:
            # Analyze the task
            analysis = await self.analyze(context)
            
            # Create detailed project plan
            project_plan = await self._create_project_plan(context, analysis)
            
            # Generate subtasks
            subtasks = await self._generate_subtasks(context, project_plan)
            
            # Create roadmap
            roadmap = await self._create_roadmap(context, project_plan, subtasks)
            
            # Store artifacts in context
            artifacts_created = []
            
            if context.blackboard:
                # Store in blackboard
                await context.blackboard.write_artifact(
                    f"project_plan_{context.work_item.id}",
                    project_plan,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"subtasks_{context.work_item.id}",
                    subtasks,
                    self.agent_id
                )
                await context.blackboard.write_artifact(
                    f"roadmap_{context.work_item.id}",
                    roadmap,
                    self.agent_id
                )
                artifacts_created = [
                    f"project_plan_{context.work_item.id}",
                    f"subtasks_{context.work_item.id}",
                    f"roadmap_{context.work_item.id}"
                ]
            else:
                # Store in context
                context.add_artifact('project_plan', project_plan, self.agent_id)
                context.add_artifact('subtasks', subtasks, self.agent_id)
                context.add_artifact('roadmap', roadmap, self.agent_id)
                artifacts_created = ['project_plan', 'subtasks', 'roadmap']
            
            # Generate insights
            insights = [
                f"Project complexity score: {analysis.get('complexity_score', 0)}/10",
                f"Identified {len(project_plan.get('phases', []))} major phases",
                f"Estimated duration: {analysis.get('estimated_duration', 'Unknown')}",
                f"Critical risks: {len(analysis.get('risks', []))} identified"
            ]
            
            # Generate recommendations
            recommendations = []
            if analysis.get('complexity_score', 0) > 7:
                recommendations.append("Consider breaking this into multiple smaller projects")
            if len(analysis.get('dependencies', [])) > 5:
                recommendations.append("High number of dependencies - careful coordination needed")
            if any('security' in risk.lower() for risk in analysis.get('risks', [])):
                recommendations.append("Security risks identified - engage SecurityAgent early")
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=True,
                output={
                    'project_plan': project_plan,
                    'subtasks': subtasks,
                    'roadmap': roadmap,
                    'analysis': analysis
                },
                artifacts_created=artifacts_created,
                insights=insights,
                recommendations=recommendations,
                confidence=0.85,
                execution_time=execution_time,
                metadata={'phases_count': len(project_plan.get('phases', []))}
            )
            
        except Exception as e:
            self.logger.error(f"PlannerAgent execution failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                success=False,
                output={'error': str(e)},
                confidence=0.0,
                execution_time=time.time() - start_time
            )
    
    async def _create_project_plan(self, context: AgentContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a detailed project plan."""
        prompt = f"""
        Create a detailed project plan for:
        Task: {context.work_item.title}
        
        Based on analysis:
        {json.dumps(analysis, indent=2)}
        
        Include:
        1. Project phases with clear objectives
        2. Deliverables for each phase
        3. Timeline with milestones
        4. Resource allocation
        5. Risk mitigation plan
        6. Communication plan
        
        Format as JSON with structure:
        {{
            "project_name": "...",
            "phases": [
                {{
                    "phase_id": "...",
                    "name": "...",
                    "objectives": [...],
                    "deliverables": [...],
                    "duration": "...",
                    "resources_needed": [...],
                    "dependencies": [...]
                }}
            ],
            "milestones": [...],
            "risk_mitigation": [...],
            "communication_plan": {{...}}
        }}
        """
        
        response = await self._call_ai_model(prompt)
        
        try:
            plan = json.loads(response)
            return plan
        except:
            # Fallback plan structure
            return {
                "project_name": context.work_item.title,
                "phases": [
                    {
                        "phase_id": "phase_1",
                        "name": "Planning & Setup",
                        "objectives": ["Define requirements", "Set up environment"],
                        "deliverables": ["Requirements document", "Development environment"],
                        "duration": "1 week",
                        "resources_needed": ["Project manager", "Technical lead"],
                        "dependencies": []
                    }
                ],
                "milestones": ["Project kickoff", "Phase 1 completion"],
                "risk_mitigation": analysis.get('risks', []),
                "communication_plan": {
                    "frequency": "Weekly",
                    "channels": ["Email", "Slack"],
                    "stakeholders": ["Team", "Management"]
                }
            }
    
    async def _generate_subtasks(self, context: AgentContext, project_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate subtasks from project plan."""
        subtasks = []
        
        for phase in project_plan.get('phases', []):
            for deliverable in phase.get('deliverables', []):
                subtask = {
                    'id': f"subtask_{len(subtasks)+1}",
                    'title': f"{phase['name']}: {deliverable}",
                    'phase': phase['name'],
                    'type': 'implementation',
                    'priority': context.work_item.priority.value,
                    'estimated_effort': phase.get('duration', '1 week'),
                    'dependencies': phase.get('dependencies', []),
                    'assigned_agent': None  # To be determined by coordinator
                }
                subtasks.append(subtask)
        
        return subtasks
    
    async def _create_roadmap(self, context: AgentContext, project_plan: Dict[str, Any], 
                            subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a visual roadmap representation."""
        roadmap = {
            'title': f"Roadmap: {project_plan.get('project_name', context.work_item.title)}",
            'timeline': {
                'start': datetime.now(timezone.utc).isoformat(),
                'phases': []
            },
            'dependencies_graph': {},
            'critical_path': []
        }
        
        current_date = datetime.now(timezone.utc)
        for i, phase in enumerate(project_plan.get('phases', [])):
            phase_entry = {
                'phase_id': phase['phase_id'],
                'name': phase['name'],
                'start': current_date.isoformat(),
                'duration': phase['duration'],
                'subtasks': [st['id'] for st in subtasks if st['phase'] == phase['name']]
            }
            roadmap['timeline']['phases'].append(phase_entry)
            
            # Build dependency graph
            for dep in phase.get('dependencies', []):
                if phase['phase_id'] not in roadmap['dependencies_graph']:
                    roadmap['dependencies_graph'][phase['phase_id']] = []
                roadmap['dependencies_graph'][phase['phase_id']].append(dep)
        
        return roadmap
    
    async def review_artifact(self, artifact_key: str, artifact_value: Any, 
                            created_by: str, context: AgentContext) -> Dict[str, Any]:
        """Review artifacts from other agents from a planning perspective."""
        review = await super().review_artifact(artifact_key, artifact_value, created_by, context)
        
        # Specific planning reviews
        if 'code' in artifact_key:
            review['feedback'].append("Check if implementation aligns with project phases")
            review['feedback'].append("Verify deliverables match planned objectives")
        elif 'test' in artifact_key:
            review['feedback'].append("Ensure test coverage aligns with success criteria")
        elif 'security' in artifact_key:
            review['feedback'].append("Verify security measures address identified risks")
        
        review['confidence'] = 0.8
        return review