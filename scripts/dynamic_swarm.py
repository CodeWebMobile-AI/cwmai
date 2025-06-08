"""
Dynamic Swarm Intelligence

Enhanced swarm that learns and adapts its behavior.
Agents have full system context and adjust based on outcomes.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import asyncio

# Import base swarm
from swarm_intelligence import RealSwarmIntelligence, RealSwarmAgent, AgentRole


class DynamicSwarmAgent(RealSwarmAgent):
    """Enhanced swarm agent with system context awareness."""
    
    def __init__(self, *args, system_context: Dict[str, Any] = None, **kwargs):
        """Initialize with system context.
        
        Args:
            system_context: Full system context including charter
            *args, **kwargs: Parent class arguments
        """
        super().__init__(*args, **kwargs)
        self.system_context = system_context or {}
        self.performance_history = []
        
    async def analyze_task(self, task: Dict[str, Any], 
                          other_insights: List[Dict[str, Any]] = None,
                          iteration: int = 1) -> Dict[str, Any]:
        """Analyze task with full system awareness.
        
        Args:
            task: Task to analyze
            other_insights: Insights from other agents
            iteration: Which iteration of analysis (1 or 2)
            
        Returns:
            Agent's analysis
        """
        # Build context-aware prompt
        prompt = self._build_contextual_prompt(task, other_insights, iteration)
        
        try:
            # Call AI model
            response = await self._call_ai_model(prompt)
            
            # Parse response
            analysis = self._parse_ai_response(response)
            analysis['agent_id'] = self.id
            analysis['agent_role'] = self.role.value
            analysis['model_used'] = self.model_name
            analysis['iteration'] = iteration
            
            # Update performance history
            self.performance_history.append({
                'task_id': task.get('id', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'analysis': analysis,
                'iteration': iteration
            })
            
            return analysis
            
        except Exception as e:
            logging.error(f"Agent {self.id} analysis failed: {e}")
            return self._error_response(str(e))
    
    def _build_contextual_prompt(self, task: Dict[str, Any],
                                other_insights: List[Dict[str, Any]] = None,
                                iteration: int = 1) -> str:
        """Build prompt with full system context.
        
        Args:
            task: Task to analyze
            other_insights: Other agents' insights
            iteration: Analysis iteration
            
        Returns:
            Contextual prompt
        """
        # Get charter and other context
        charter = self.system_context.get('charter', {})
        active_projects = self.system_context.get('active_projects', [])
        recent_outcomes = self.system_context.get('recent_outcomes', [])
        success_patterns = self.system_context.get('success_patterns', {})
        
        # Build base prompt
        prompt = f"""
        You are a {self.role.value} expert in an AI Development Orchestrator system.
        {self.persona}
        
        SYSTEM CONTEXT:
        Primary Purpose: {charter.get('PRIMARY_PURPOSE', 'Build software portfolio')}
        Core Objectives: {json.dumps(charter.get('CORE_OBJECTIVES', []), indent=2)}
        
        Active Projects in Portfolio:
        {json.dumps(active_projects, indent=2)}
        
        Recent Outcomes (for learning):
        {json.dumps(recent_outcomes[-3:], indent=2)}
        
        Success Patterns:
        {json.dumps(success_patterns, indent=2)}
        
        CRITICAL RULES:
        1. NEW_PROJECT tasks must describe complete applications using Laravel React starter kit
        2. FEATURE tasks must target specific existing projects from the portfolio above
        3. improvement tasks enhance the AI orchestrator system itself
        4. Never suggest features for non-existent projects
        5. All decisions must align with the system charter
        """
        
        # Add iteration-specific context
        if iteration == 1:
            prompt += f"""
        
        ITERATION 1 - INDEPENDENT ANALYSIS:
        Analyze this task from your {self.role.value} perspective:
        
        Task Details:
        {json.dumps(task, indent=2)}
        
        Provide your independent analysis including:
        1. Key insights from your {self.role.value} expertise
        2. Potential challenges or risks you foresee
        3. Specific recommendations for implementation
        4. Priority level from your viewpoint (1-10)
        5. Estimated effort/complexity
        6. Alignment with system purpose
        """
        else:  # iteration == 2
            prompt += f"""
        
        ITERATION 2 - COLLABORATIVE REFINEMENT:
        Refine your analysis considering other agents' perspectives:
        
        Task Details:
        {json.dumps(task, indent=2)}
        
        Other Agents' Insights:
        {self._format_other_insights(other_insights)}
        
        Refine your analysis by:
        1. Incorporating valuable insights from other agents
        2. Resolving any conflicts in perspectives
        3. Strengthening your recommendations
        4. Adjusting priority/complexity based on collective wisdom
        5. Identifying cross-functional considerations
        6. Building consensus while maintaining your expertise
        """
        
        prompt += """
        
        Format your response as JSON with:
        - key_points: List of main insights
        - challenges: List of identified challenges
        - recommendations: List of specific recommendations
        - priority: Your priority score (1-10)
        - complexity: 'low', 'medium', or 'high'
        - confidence: Your confidence level (0-1)
        - alignment_score: How well this aligns with system purpose (0-1)
        - cross_functional_notes: Considerations for other roles (iteration 2 only)
        """
        
        return prompt
    
    def _format_other_insights(self, other_insights: List[Dict[str, Any]]) -> str:
        """Format other agents' insights for prompt.
        
        Args:
            other_insights: List of other agents' analyses
            
        Returns:
            Formatted string
        """
        if not other_insights:
            return "No other insights available yet."
            
        formatted = []
        for insight in other_insights:
            agent_role = insight.get('agent_role', 'Unknown')
            key_points = insight.get('key_points', [])
            priority = insight.get('priority', 'N/A')
            
            formatted.append(f"""
            {agent_role.upper()} Agent:
            - Key Points: {'; '.join(key_points[:3])}
            - Priority: {priority}
            - Main Challenge: {insight.get('challenges', ['None'])[0]}
            """)
            
        return '\n'.join(formatted)
    
    def _error_response(self, error: str) -> Dict[str, Any]:
        """Generate error response.
        
        Args:
            error: Error message
            
        Returns:
            Error response dict
        """
        return {
            'agent_id': self.id,
            'agent_role': self.role.value,
            'error': error,
            'key_points': [],
            'recommendations': [],
            'priority': 5,
            'complexity': 'unknown',
            'confidence': 0,
            'alignment_score': 0
        }
    
    def update_context(self, new_context: Dict[str, Any]) -> None:
        """Update agent's system context.
        
        Args:
            new_context: Updated context
        """
        self.system_context.update(new_context)


class DynamicSwarmIntelligence(RealSwarmIntelligence):
    """Enhanced swarm that learns and adapts based on outcomes."""
    
    def __init__(self, ai_brain, learning_system=None, charter_system=None):
        """Initialize dynamic swarm.
        
        Args:
            ai_brain: AI brain for agents
            learning_system: Outcome learning system
            charter_system: Dynamic charter system
        """
        super().__init__(ai_brain)
        self.learning_system = learning_system
        self.charter_system = charter_system
        self.swarm_history = []
        self.agent_performance_tracking = {}
        
        # Convert agents to dynamic agents
        self._convert_to_dynamic_agents()
        
    def _convert_to_dynamic_agents(self) -> None:
        """Convert base agents to dynamic agents."""
        dynamic_agents = []
        
        for agent in self.agents:
            dynamic_agent = DynamicSwarmAgent(
                id=agent.id,
                role=agent.role,
                model_name=agent.model_name,
                expertise_areas=agent.expertise_areas,
                persona=agent.persona,
                ai_brain=agent.ai_brain,
                task_history=agent.task_history,
                system_context={}
            )
            dynamic_agents.append(dynamic_agent)
            
        self.agents = dynamic_agents
    
    async def process_task_swarm(self, task: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process task with context-aware dynamic swarm.
        
        Args:
            task: Task to analyze
            context: Full system context
            
        Returns:
            Swarm analysis result
        """
        start_time = datetime.now(timezone.utc)
        context = context or {}
        
        # Get current charter
        if self.charter_system:
            charter = await self.charter_system.get_current_charter()
            context['charter'] = charter
            
        # Get success patterns from learning
        if self.learning_system:
            learning_summary = self.learning_system.get_learning_summary()
            context['success_patterns'] = learning_summary.get('value_patterns', {})
            context['recent_outcomes'] = learning_summary.get('recent_outcomes', [])
            
        # Update all agents with context
        for agent in self.agents:
            agent.update_context(context)
            
        # Run enhanced swarm process
        result = await self._enhanced_swarm_process(task, context)
        
        # Record in history
        self._record_swarm_analysis(task, result, context)
        
        # Update metrics
        end_time = datetime.now(timezone.utc)
        result['duration_seconds'] = (end_time - start_time).total_seconds()
        self._update_swarm_metrics(result)
        
        return result
    
    async def _enhanced_swarm_process(self, task: Dict[str, Any],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced swarm process with learning integration.
        
        Args:
            task: Task to analyze
            context: System context
            
        Returns:
            Enhanced swarm result
        """
        # Phase 1: Independent Analysis
        individual_analyses = await self._phase_individual_analysis(task)
        
        # Phase 2: Cross-Pollination with Learning
        refined_analyses = await self._phase_enhanced_cross_pollination(
            task, individual_analyses, context
        )
        
        # Phase 3: Intelligent Consensus
        consensus = await self._phase_intelligent_consensus(
            refined_analyses, context
        )
        
        # Phase 4: Strategic Action Planning
        action_plan = await self._phase_strategic_planning(
            task, consensus, context
        )
        
        # Phase 5: Value Prediction
        value_prediction = await self._predict_outcome_value(
            task, consensus, action_plan
        )
        
        return {
            'task_id': task.get('id', 'unknown'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'individual_analyses': individual_analyses,
            'refined_analyses': refined_analyses,
            'consensus': consensus,
            'action_plan': action_plan,
            'value_prediction': value_prediction,
            'collective_review': self._generate_enhanced_review(
                consensus, action_plan, value_prediction
            )
        }
    
    async def _phase_enhanced_cross_pollination(self, task: Dict[str, Any],
                                               initial_analyses: List[Dict[str, Any]],
                                               context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced cross-pollination with learning insights.
        
        Args:
            task: Task being analyzed
            initial_analyses: First iteration analyses
            context: System context
            
        Returns:
            Refined analyses
        """
        # Add learning insights to cross-pollination
        if self.learning_system:
            # Get insights about similar tasks
            similar_outcomes = self.learning_system._find_similar_outcomes(task)
            context['similar_task_outcomes'] = similar_outcomes
            
        # Each agent refines with full context
        refined_analyses = []
        
        for i, agent in enumerate(self.agents):
            # Get other agents' analyses
            other_analyses = [a for j, a in enumerate(initial_analyses) if j != i]
            
            # Refine with iteration 2
            refined = await agent.analyze_task(task, other_analyses, iteration=2)
            refined_analyses.append(refined)
            
        return refined_analyses
    
    async def _phase_intelligent_consensus(self, analyses: List[Dict[str, Any]],
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Build intelligent consensus using orchestrator and learning.
        
        Args:
            analyses: All agents' analyses
            context: System context
            
        Returns:
            Intelligent consensus
        """
        # Extract all insights
        all_key_points = []
        all_challenges = []
        all_recommendations = []
        priorities = []
        alignment_scores = []
        
        for analysis in analyses:
            if 'error' not in analysis:
                all_key_points.extend(analysis.get('key_points', []))
                all_challenges.extend(analysis.get('challenges', []))
                all_recommendations.extend(analysis.get('recommendations', []))
                priorities.append(analysis.get('priority', 5))
                alignment_scores.append(analysis.get('alignment_score', 0.5))
                
        # Use orchestrator to synthesize
        orchestrator = next((a for a in self.agents if a.role == AgentRole.ORCHESTRATOR), None)
        
        if orchestrator and self.ai_brain:
            synthesis_prompt = f"""
            As the swarm orchestrator, create an intelligent consensus from all analyses.
            
            System Charter: {json.dumps(context.get('charter', {}), indent=2)}
            
            All Agents' Insights:
            - Key Points: {json.dumps(all_key_points, indent=2)}
            - Challenges: {json.dumps(all_challenges, indent=2)}
            - Recommendations: {json.dumps(all_recommendations, indent=2)}
            - Priority Scores: {priorities}
            - Alignment Scores: {alignment_scores}
            
            Similar Past Task Outcomes:
            {json.dumps(context.get('similar_task_outcomes', []), indent=2)}
            
            Create a consensus that:
            1. Identifies the most important insights (top 5)
            2. Prioritizes critical challenges (top 3)
            3. Selects best recommendations (top 5)
            4. Resolves conflicting viewpoints with reasoning
            5. Provides unified priority and alignment scores
            6. Considers lessons from similar past tasks
            7. Ensures alignment with system charter
            
            Format as JSON with:
            - key_insights: Top 5 insights with rationale
            - critical_challenges: Top 3 challenges with severity
            - top_recommendations: Top 5 actionable recommendations
            - conflicts_resolved: How conflicts were resolved
            - consensus_priority: Unified priority (1-10)
            - consensus_alignment: Alignment with charter (0-1)
            - success_probability: Estimated chance of success (0-1)
            - strategic_value: Long-term value assessment
            """
            
            response = await orchestrator._call_ai_model(synthesis_prompt)
            consensus = orchestrator._parse_ai_response(response)
            
            # Add computed metrics
            consensus['average_priority'] = sum(priorities) / len(priorities) if priorities else 5
            consensus['average_alignment'] = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
            
        else:
            # Fallback consensus
            consensus = self._create_basic_consensus(
                all_key_points, all_challenges, all_recommendations,
                priorities, alignment_scores
            )
            
        return consensus
    
    async def _phase_strategic_planning(self, task: Dict[str, Any],
                                       consensus: Dict[str, Any],
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Create strategic action plan based on consensus and learning.
        
        Args:
            task: Task being analyzed
            consensus: Swarm consensus
            context: System context
            
        Returns:
            Strategic action plan
        """
        strategist = next((a for a in self.agents if a.role == AgentRole.STRATEGIST), None)
        architect = next((a for a in self.agents if a.role == AgentRole.ARCHITECT), None)
        
        if strategist and self.ai_brain:
            planning_prompt = f"""
            Create a strategic action plan based on swarm consensus.
            
            Task: {json.dumps(task, indent=2)}
            
            Swarm Consensus:
            {json.dumps(consensus, indent=2)}
            
            System Context:
            - Active Projects: {len(context.get('active_projects', []))}
            - Success Patterns: {json.dumps(
                self.learning_system.get_learning_summary().get('high_value_task_types', [])
                if self.learning_system else []
            )}
            
            Create a strategic plan with:
            1. Immediate actions (next 24-48 hours)
            2. Milestone targets (weekly goals)
            3. Success metrics (measurable outcomes)
            4. Risk mitigation strategies
            5. Resource requirements
            6. Integration points with existing projects
            7. Long-term impact assessment
            
            For NEW_PROJECT tasks: Include Laravel React starter kit setup steps
            For FEATURE tasks: Include integration with target project
            For improvement tasks: Include testing and rollback plans
            
            Format as JSON with clear, actionable items.
            """
            
            response = await strategist._call_ai_model(planning_prompt)
            action_plan = strategist._parse_ai_response(response)
            
        else:
            action_plan = self._create_basic_action_plan(task, consensus)
            
        return action_plan
    
    async def _predict_outcome_value(self, task: Dict[str, Any],
                                    consensus: Dict[str, Any],
                                    action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the value this task will create.
        
        Args:
            task: Task being analyzed
            consensus: Swarm consensus
            action_plan: Action plan
            
        Returns:
            Value prediction
        """
        if self.learning_system:
            # Get prediction from learning system
            prediction = await self.learning_system.predict_task_value(task)
            
            # Enhance with swarm insights
            prediction['swarm_confidence'] = consensus.get('success_probability', 0.5)
            prediction['strategic_alignment'] = consensus.get('consensus_alignment', 0.5)
            
            return prediction
        else:
            # Basic prediction based on consensus
            return {
                'predicted_value': consensus.get('strategic_value', 0.5),
                'success_probability': consensus.get('success_probability', 0.5),
                'confidence': 0.7,
                'reasoning': 'Based on swarm consensus without historical data'
            }
    
    def _generate_enhanced_review(self, consensus: Dict[str, Any],
                                 action_plan: Dict[str, Any],
                                 value_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced collective review.
        
        Args:
            consensus: Swarm consensus
            action_plan: Action plan
            value_prediction: Predicted value
            
        Returns:
            Enhanced review
        """
        return {
            'summary': f"Dynamic swarm analyzed task with {len(self.agents)} AI agents",
            'consensus_priority': consensus.get('consensus_priority', 5),
            'alignment_score': consensus.get('consensus_alignment', 0.5),
            'success_probability': consensus.get('success_probability', 0.5),
            'predicted_value': value_prediction.get('predicted_value', 0.5),
            'key_insights': consensus.get('key_insights', [])[:3],
            'immediate_actions': action_plan.get('immediate_actions', [])[:3],
            'confidence_level': self._calculate_swarm_confidence(consensus, value_prediction),
            'top_suggestions': self._extract_top_suggestions(consensus, action_plan),
            'recommendation': self._make_recommendation(consensus, value_prediction)
        }
    
    def _make_recommendation(self, consensus: Dict[str, Any],
                           value_prediction: Dict[str, Any]) -> str:
        """Make execution recommendation based on analysis.
        
        Args:
            consensus: Swarm consensus
            value_prediction: Predicted value
            
        Returns:
            Recommendation
        """
        alignment = consensus.get('consensus_alignment', 0.5)
        priority = consensus.get('consensus_priority', 5)
        predicted_value = value_prediction.get('predicted_value', 0.5)
        success_prob = consensus.get('success_probability', 0.5)
        
        # Decision logic
        if alignment < 0.3:
            return "SKIP - Poor alignment with system charter"
        elif predicted_value < 0.3 and success_prob < 0.5:
            return "SKIP - Low value and success probability"
        elif priority >= 8 and alignment >= 0.7:
            return "EXECUTE_IMMEDIATELY - High priority and alignment"
        elif predicted_value >= 0.7:
            return "EXECUTE - High predicted value"
        elif priority >= 6:
            return "EXECUTE - Good priority"
        else:
            return "CONSIDER - Moderate value, reassess if needed"
    
    def _record_swarm_analysis(self, task: Dict[str, Any],
                              result: Dict[str, Any],
                              context: Dict[str, Any]) -> None:
        """Record swarm analysis for learning.
        
        Args:
            task: Analyzed task
            result: Swarm result
            context: System context
        """
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task': task,
            'result': result,
            'context_snapshot': {
                'project_count': len(context.get('active_projects', [])),
                'charter_version': context.get('charter', {}).get('timestamp', 'unknown')
            }
        }
        
        self.swarm_history.append(record)
        
        # Track agent performance
        for analysis in result.get('refined_analyses', []):
            agent_id = analysis.get('agent_id')
            if agent_id:
                if agent_id not in self.agent_performance_tracking:
                    self.agent_performance_tracking[agent_id] = []
                    
                self.agent_performance_tracking[agent_id].append({
                    'timestamp': record['timestamp'],
                    'confidence': analysis.get('confidence', 0),
                    'alignment': analysis.get('alignment_score', 0)
                })
    
    def _create_basic_consensus(self, key_points: List[str],
                               challenges: List[str],
                               recommendations: List[str],
                               priorities: List[float],
                               alignments: List[float]) -> Dict[str, Any]:
        """Create basic consensus when orchestrator unavailable.
        
        Args:
            key_points: All key points
            challenges: All challenges
            recommendations: All recommendations
            priorities: Priority scores
            alignments: Alignment scores
            
        Returns:
            Basic consensus
        """
        import numpy as np
        
        return {
            'key_insights': list(set(key_points))[:5],
            'critical_challenges': list(set(challenges))[:3],
            'top_recommendations': list(set(recommendations))[:5],
            'consensus_priority': int(np.mean(priorities)) if priorities else 5,
            'consensus_alignment': np.mean(alignments) if alignments else 0.5,
            'success_probability': 0.6,  # Default moderate probability
            'strategic_value': 0.5  # Default moderate value
        }
    
    def _create_basic_action_plan(self, task: Dict[str, Any],
                                 consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic action plan when strategist unavailable.
        
        Args:
            task: Task being analyzed
            consensus: Swarm consensus
            
        Returns:
            Basic action plan
        """
        return {
            'immediate_actions': consensus.get('top_recommendations', [])[:3],
            'milestone_targets': ['Complete initial implementation', 'Test thoroughly', 'Deploy'],
            'success_metrics': ['Task completed successfully', 'No critical issues'],
            'risk_mitigation': ['Regular testing', 'Incremental deployment'],
            'resource_requirements': ['Development time', 'Testing resources']
        }
    
    def get_swarm_analytics(self) -> Dict[str, Any]:
        """Get analytics about swarm performance.
        
        Returns:
            Swarm analytics
        """
        analytics = {
            'total_analyses': len(self.swarm_history),
            'agent_count': len(self.agents),
            'agent_roles': [a.role.value for a in self.agents],
            'performance_metrics': self.swarm_performance_metrics,
            'agent_performance': self._analyze_agent_performance()
        }
        
        if self.swarm_history:
            recent = self.swarm_history[-10:]
            analytics['recent_recommendations'] = [
                r['result']['collective_review']['recommendation']
                for r in recent
                if 'collective_review' in r['result']
            ]
            
        return analytics
    
    def _analyze_agent_performance(self) -> Dict[str, Any]:
        """Analyze individual agent performance.
        
        Returns:
            Agent performance analysis
        """
        performance = {}
        
        for agent_id, history in self.agent_performance_tracking.items():
            if history:
                recent = history[-20:]  # Last 20 analyses
                performance[agent_id] = {
                    'average_confidence': sum(h['confidence'] for h in recent) / len(recent),
                    'average_alignment': sum(h['alignment'] for h in recent) / len(recent),
                    'total_analyses': len(history)
                }
                
        return performance