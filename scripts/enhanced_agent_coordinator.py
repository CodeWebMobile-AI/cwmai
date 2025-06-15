"""
Enhanced Agent Coordinator for Collaborative Multi-Agent Systems

Orchestrates specialized agents alongside existing swarm intelligence,
managing their collaboration through shared blackboard workspace.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict

from agent_factory import AgentFactory
from base_agent import AgentContext, AgentResult
from swarm_intelligence import RealSwarmIntelligence
from work_item_types import WorkItem, TaskPriority


class EnhancedAgentCoordinator:
    """Coordinates specialized agents with existing swarm intelligence."""
    
    def __init__(self, ai_brain: Optional[Any] = None, swarm_intelligence: Optional[RealSwarmIntelligence] = None):
        """Initialize the enhanced coordinator."""
        self.ai_brain = ai_brain
        self.swarm_intelligence = swarm_intelligence
        self.agent_factory = AgentFactory(ai_brain)
        self.blackboard = None  # Will be set when blackboard is created
        self.logger = logging.getLogger(f"{__name__}.EnhancedAgentCoordinator")
        
        # Coordination metrics
        self.coordination_metrics = {
            'tasks_coordinated': 0,
            'agent_collaborations': 0,
            'consensus_reached': 0,
            'conflicts_resolved': 0,
            'avg_coordination_time': 0.0
        }
        
    def set_blackboard(self, blackboard: 'BlackboardWorkspace') -> None:
        """Set the blackboard workspace for agent collaboration."""
        self.blackboard = blackboard
        
    async def coordinate_work_item(self, work_item: WorkItem) -> Dict[str, Any]:
        """Coordinate both specialized agents and swarm intelligence for a work item."""
        start_time = time.time()
        coordination_result = {
            'work_item_id': work_item.id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'specialized_agent_results': {},
            'swarm_analysis': None,
            'consensus': None,
            'final_recommendation': None,
            'coordination_time': 0.0
        }
        
        try:
            # Phase 1: Get swarm intelligence analysis if available
            if self.swarm_intelligence:
                self.logger.info(f"Getting swarm analysis for work item: {work_item.id}")
                swarm_task = {
                    'id': work_item.id,
                    'type': work_item.task_type,
                    'title': work_item.title,
                    'description': work_item.description,
                    'priority': work_item.priority.name,
                    'requirements': work_item.metadata.get('requirements', [])
                }
                swarm_analysis = await self.swarm_intelligence.process_task_swarm(swarm_task)
                coordination_result['swarm_analysis'] = swarm_analysis
            
            # Phase 2: Create specialized agent team
            recommendations = self.agent_factory.get_agent_recommendations(work_item)
            agent_team = []
            
            # Create recommended agents
            for agent_type in recommendations['primary_agents']:
                agent = self.agent_factory.create_agent(agent_type)
                agent_team.append(agent)
            
            for agent_type in recommendations['supporting_agents'][:2]:  # Limit supporting agents
                agent = self.agent_factory.create_agent(agent_type)
                agent_team.append(agent)
            
            self.logger.info(f"Created agent team: {[a.agent_type for a in agent_team]}")
            
            # Phase 3: Execute specialized agents in parallel
            agent_results = await self._execute_agent_team(work_item, agent_team)
            coordination_result['specialized_agent_results'] = {
                result.agent_type: result.to_dict() for result in agent_results
            }
            
            # Phase 4: Enable agent collaboration and review
            collaborative_results = await self._enable_agent_collaboration(
                work_item, agent_team, agent_results
            )
            
            # Phase 5: Build consensus from all sources
            consensus = await self._build_comprehensive_consensus(
                work_item, agent_results, collaborative_results, 
                coordination_result.get('swarm_analysis')
            )
            coordination_result['consensus'] = consensus
            
            # Phase 6: Generate final recommendation
            final_recommendation = await self._generate_final_recommendation(
                work_item, consensus, agent_results
            )
            coordination_result['final_recommendation'] = final_recommendation
            
            # Update metrics
            coordination_time = time.time() - start_time
            coordination_result['coordination_time'] = coordination_time
            self._update_coordination_metrics(coordination_result)
            
            return coordination_result
            
        except Exception as e:
            self.logger.error(f"Coordination failed for work item {work_item.id}: {e}")
            coordination_result['error'] = str(e)
            coordination_result['coordination_time'] = time.time() - start_time
            return coordination_result
    
    async def _execute_agent_team(self, work_item: WorkItem, agent_team: List['BaseAgent']) -> List[AgentResult]:
        """Execute all agents in the team in parallel."""
        # Create shared context
        context = AgentContext(
            work_item=work_item,
            blackboard=self.blackboard,
            other_agents=agent_team,
            shared_artifacts={},
            execution_history=[]
        )
        
        # Execute agents in parallel
        async_tasks = []
        for agent in agent_team:
            async_tasks.append(agent.execute(context))
        
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Handle exceptions and update performance
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Agent {agent_team[i].agent_id} failed: {result}")
                # Create failure result
                failure_result = AgentResult(
                    agent_id=agent_team[i].agent_id,
                    agent_type=agent_team[i].agent_type,
                    success=False,
                    output={'error': str(result)},
                    confidence=0.0,
                    execution_time=0.0
                )
                processed_results.append(failure_result)
            else:
                processed_results.append(result)
                # Update agent performance
                self.agent_factory.update_agent_performance(result.agent_id, result)
        
        return processed_results
    
    async def _enable_agent_collaboration(self, work_item: WorkItem, agent_team: List['BaseAgent'], 
                                        initial_results: List[AgentResult]) -> List[AgentResult]:
        """Enable agents to collaborate and review each other's work."""
        collaborative_results = []
        
        # Create collaboration context with all initial results
        context = AgentContext(
            work_item=work_item,
            blackboard=self.blackboard,
            other_agents=agent_team,
            shared_artifacts={},
            execution_history=[r.to_dict() for r in initial_results]
        )
        
        # Let each agent review others' artifacts
        for agent in agent_team:
            # Get other agents' results
            other_results = [r for r in initial_results if r.agent_id != agent.agent_id]
            
            # Agent collaborates based on others' work
            collab_result = await agent.collaborate(context, other_results)
            collaborative_results.append(collab_result)
            
            # Perform artifact reviews if blackboard available
            if self.blackboard:
                for result in other_results:
                    for artifact_key in result.artifacts_created:
                        artifact_value = await self.blackboard.read_artifact(artifact_key)
                        if artifact_value:
                            review = await agent.review_artifact(
                                artifact_key, artifact_value['data'], 
                                result.agent_id, context
                            )
                            # Store review
                            await self.blackboard.add_review(artifact_key, review)
        
        self.coordination_metrics['agent_collaborations'] += len(collaborative_results)
        return collaborative_results
    
    async def _build_comprehensive_consensus(self, work_item: WorkItem, 
                                           agent_results: List[AgentResult],
                                           collaborative_results: List[AgentResult],
                                           swarm_analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build consensus from all agent and swarm inputs."""
        consensus = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'work_item_id': work_item.id,
            'key_agreements': [],
            'conflicts': [],
            'unified_recommendations': [],
            'priority_actions': [],
            'confidence_level': 0.0,
            'consensus_type': 'hybrid'  # hybrid of specialized agents and swarm
        }
        
        # Collect all insights and recommendations
        all_insights = []
        all_recommendations = []
        
        # From specialized agents
        for result in agent_results + collaborative_results:
            if result.success:
                all_insights.extend([
                    {'source': result.agent_type, 'insight': insight, 'confidence': result.confidence}
                    for insight in result.insights
                ])
                all_recommendations.extend([
                    {'source': result.agent_type, 'recommendation': rec, 'confidence': result.confidence}
                    for rec in result.recommendations
                ])
        
        # From swarm intelligence
        if swarm_analysis and swarm_analysis.get('consensus'):
            swarm_consensus = swarm_analysis['consensus']
            all_insights.extend([
                {'source': 'swarm', 'insight': insight, 'confidence': 0.8}
                for insight in swarm_consensus.get('key_insights', [])
            ])
            all_recommendations.extend([
                {'source': 'swarm', 'recommendation': rec, 'confidence': 0.8}
                for rec in swarm_consensus.get('top_recommendations', [])
            ])
        
        # Find agreements (insights/recommendations from multiple sources)
        insight_counts = defaultdict(list)
        for item in all_insights:
            insight_counts[item['insight']].append(item['source'])
        
        rec_counts = defaultdict(list)
        for item in all_recommendations:
            rec_counts[item['recommendation']].append(item['source'])
        
        # Key agreements are insights from 2+ sources
        consensus['key_agreements'] = [
            {
                'insight': insight,
                'sources': sources,
                'agreement_strength': len(sources) / len(agent_results)
            }
            for insight, sources in insight_counts.items()
            if len(sources) >= 2
        ]
        
        # Unified recommendations
        consensus['unified_recommendations'] = [
            {
                'recommendation': rec,
                'sources': sources,
                'priority': 'high' if len(sources) >= 3 else 'medium'
            }
            for rec, sources in rec_counts.items()
        ]
        
        # Identify conflicts
        consensus['conflicts'] = self._identify_conflicts(agent_results, swarm_analysis)
        
        # Priority actions based on agent types
        security_critical = any(r.agent_type == 'security' and len(r.recommendations) > 0 
                              for r in agent_results if r.success)
        if security_critical:
            consensus['priority_actions'].append({
                'action': 'Address security recommendations before proceeding',
                'priority': 'critical'
            })
        
        # Calculate overall confidence
        confidences = [r.confidence for r in agent_results if r.success]
        consensus['confidence_level'] = sum(confidences) / len(confidences) if confidences else 0.0
        
        self.coordination_metrics['consensus_reached'] += 1
        return consensus
    
    def _identify_conflicts(self, agent_results: List[AgentResult], 
                          swarm_analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify conflicts between different agents or swarm analysis."""
        conflicts = []
        
        # Check for conflicting recommendations
        recommendations_by_topic = defaultdict(list)
        
        for result in agent_results:
            if result.success:
                for rec in result.recommendations:
                    # Simple topic extraction (could be enhanced)
                    if 'test' in rec.lower():
                        topic = 'testing'
                    elif 'security' in rec.lower():
                        topic = 'security'
                    elif 'refactor' in rec.lower():
                        topic = 'refactoring'
                    else:
                        topic = 'general'
                    
                    recommendations_by_topic[topic].append({
                        'source': result.agent_type,
                        'recommendation': rec
                    })
        
        # Detect potential conflicts (simplified)
        for topic, recs in recommendations_by_topic.items():
            if len(recs) > 1:
                # Check for contradictions
                if any('not' in r['recommendation'].lower() for r in recs) and \
                   any('should' in r['recommendation'].lower() for r in recs):
                    conflicts.append({
                        'type': 'recommendation_conflict',
                        'topic': topic,
                        'conflicting_sources': [r['source'] for r in recs],
                        'details': recs
                    })
        
        self.coordination_metrics['conflicts_resolved'] += len(conflicts)
        return conflicts
    
    async def _generate_final_recommendation(self, work_item: WorkItem, consensus: Dict[str, Any],
                                           agent_results: List[AgentResult]) -> Dict[str, Any]:
        """Generate final coordinated recommendation."""
        recommendation = {
            'summary': f"Coordinated analysis for {work_item.title}",
            'confidence': consensus['confidence_level'],
            'next_steps': [],
            'implementation_order': [],
            'estimated_effort': None,
            'risks': [],
            'dependencies': []
        }
        
        # Determine implementation order based on agent results
        if any(r.agent_type == 'planner' and r.success for r in agent_results):
            planner_result = next(r for r in agent_results if r.agent_type == 'planner' and r.success)
            if 'subtasks' in planner_result.output:
                recommendation['implementation_order'] = [
                    st['title'] for st in planner_result.output['subtasks'][:5]
                ]
        
        # Aggregate next steps from consensus
        for rec in consensus['unified_recommendations'][:5]:
            recommendation['next_steps'].append({
                'action': rec['recommendation'],
                'priority': rec['priority'],
                'supported_by': rec['sources']
            })
        
        # Add priority actions
        recommendation['next_steps'].extend([
            {'action': pa['action'], 'priority': pa['priority'], 'supported_by': ['coordinator']}
            for pa in consensus['priority_actions']
        ])
        
        # Extract risks from security agent
        security_results = [r for r in agent_results if r.agent_type == 'security' and r.success]
        if security_results:
            security_output = security_results[0].output
            if 'security_audit' in security_output:
                risks = security_output['security_audit'].get('risks', [])
                recommendation['risks'] = risks[:3]  # Top 3 risks
        
        return recommendation
    
    def _update_coordination_metrics(self, coordination_result: Dict[str, Any]) -> None:
        """Update coordination metrics."""
        self.coordination_metrics['tasks_coordinated'] += 1
        
        # Update average coordination time
        n = self.coordination_metrics['tasks_coordinated']
        avg_time = self.coordination_metrics['avg_coordination_time']
        new_time = coordination_result['coordination_time']
        self.coordination_metrics['avg_coordination_time'] = (avg_time * (n-1) + new_time) / n
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status and metrics."""
        factory_status = self.agent_factory.get_factory_status()
        
        return {
            'coordinator_metrics': self.coordination_metrics,
            'factory_status': factory_status,
            'swarm_available': self.swarm_intelligence is not None,
            'blackboard_available': self.blackboard is not None,
            'coordination_mode': 'hybrid' if self.swarm_intelligence else 'specialized_only'
        }