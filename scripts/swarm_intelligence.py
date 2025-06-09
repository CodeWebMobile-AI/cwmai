"""
Real Swarm Intelligence System

Implements multi-agent coordination using actual AI models for parallel task execution and emergent intelligence.
Each agent is powered by a real AI model with specific expertise and persona.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timezone
import logging


class AgentRole(Enum):
    """Specialized roles for swarm agents."""
    ARCHITECT = "architect"          # System design and architecture
    DEVELOPER = "developer"          # Code implementation
    TESTER = "tester"               # Testing and quality assurance
    SECURITY = "security"           # Security analysis
    PERFORMANCE = "performance"     # Performance optimization
    RESEARCHER = "researcher"       # Market and tech research
    STRATEGIST = "strategist"       # Business strategy
    LEARNER = "learner"            # ML model training
    REVIEWER = "reviewer"          # Code review specialist
    ORCHESTRATOR = "orchestrator"  # Swarm coordinator


@dataclass
class RealSwarmAgent:
    """Individual AI-powered agent in the swarm."""
    id: str
    role: AgentRole
    model_name: str  # claude-3-opus, gpt-4, gemini-pro, etc.
    expertise_areas: List[str]
    persona: str  # Agent's personality and approach
    ai_brain: Any  # Reference to AI brain for making real calls
    task_history: List[Dict[str, Any]]
    performance_score: float = 1.0
    
    async def analyze_task(self, task: Dict[str, Any], 
                          other_insights: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Have this agent analyze a task using its AI model."""
        # Build context from other agents' insights
        context = ""
        if other_insights:
            context = "\n\nOther agents' insights:\n"
            for insight in other_insights:
                context += f"- {insight['agent_role']}: {insight['key_points']}\n"
        
        # Create agent-specific prompt
        prompt = f"""
        You are a {self.role.value} expert with the following expertise: {', '.join(self.expertise_areas)}.
        {self.persona}
        
        Analyze this task from your specialized perspective:
        Task Type: {task.get('type', 'unknown')}
        Title: {task.get('title', 'untitled')}
        Description: {task.get('description', 'no description')}
        Requirements: {json.dumps(task.get('requirements', []))}
        {context}
        
        Provide your analysis including:
        1. Key insights from your {self.role.value} perspective
        2. Potential challenges or risks you foresee
        3. Specific recommendations for implementation
        4. Priority level from your viewpoint (1-10)
        5. Estimated effort/complexity
        
        Format your response as a JSON object with keys:
        - key_points: List of your main insights
        - challenges: List of identified challenges
        - recommendations: List of specific recommendations
        - priority: Your priority score (1-10)
        - complexity: low/medium/high
        - confidence: Your confidence in this analysis (0-1)
        """
        
        try:
            # Use the specific model for this agent
            response = await self._call_ai_model(prompt)
            
            # Parse response
            analysis = self._parse_ai_response(response)
            analysis['agent_id'] = self.id
            analysis['agent_role'] = self.role.value
            analysis['model_used'] = self.model_name
            
            # Update task history
            self.task_history.append({
                'task_id': task.get('id', 'unknown'),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'analysis': analysis
            })
            
            return analysis
            
        except Exception as e:
            logging.error(f"Agent {self.id} analysis failed: {e}")
            return {
                'agent_id': self.id,
                'agent_role': self.role.value,
                'error': str(e),
                'key_points': [],
                'recommendations': [],
                'priority': 5,
                'complexity': 'unknown'
            }
    
    async def _call_ai_model(self, prompt: str) -> str:
        """Call the AI model based on agent's configuration."""
        if not self.ai_brain:
            return "No AI brain configured"
            
        try:
            # Route to appropriate model - properly await async calls
            if 'claude' in self.model_name.lower():
                response = await self.ai_brain.generate_enhanced_response(prompt, model='claude')
            elif 'gpt' in self.model_name.lower():
                response = await self.ai_brain.generate_enhanced_response(prompt, model='gpt')
            elif 'gemini' in self.model_name.lower():
                response = await self.ai_brain.generate_enhanced_response(prompt, model='gemini')
            else:
                # Default to any available model
                response = await self.ai_brain.generate_enhanced_response(prompt)
                
            return response.get('content', '') if response else ""
        except Exception as e:
            print(f"Error calling AI model {self.model_name}: {e}")
            return f"Error generating response: {str(e)}"
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured format with comprehensive logging."""
        import re
        import json
        
        # Add logging for response parsing
        print(f"[SWARM_PARSE] Agent {self.id} parsing response (length: {len(response)})")
        
        # Try to extract JSON from response
        try:
            # FIXED: Use proper nested JSON regex instead of broken r'\{[^{}]*\}'
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                print(f"[SWARM_PARSE] Agent {self.id} extracted JSON: {json_str[:100]}...")
                
                # FIXED: Use json.loads() instead of ast.literal_eval()
                parsed_data = json.loads(json_str)
                
                # Validate required fields and log missing ones
                required_fields = ['key_points', 'challenges', 'recommendations', 'priority']
                for field in required_fields:
                    if field not in parsed_data:
                        print(f"[SWARM_PARSE] WARNING: Agent {self.id} missing required field '{field}'")
                        if field in ['key_points', 'challenges', 'recommendations']:
                            parsed_data[field] = []
                        elif field == 'priority':
                            parsed_data[field] = 5
                    elif field in ['key_points', 'challenges', 'recommendations'] and not isinstance(parsed_data[field], list):
                        print(f"[SWARM_PARSE] WARNING: Agent {self.id} field '{field}' is not a list, converting")
                        parsed_data[field] = []
                
                # Log if challenges list is empty (this causes the line 194 crash)
                if not parsed_data.get('challenges'):
                    print(f"[SWARM_PARSE] WARNING: Agent {self.id} ({self.role.value}) produced EMPTY challenges list!")
                
                print(f"[SWARM_PARSE] Agent {self.id} successfully parsed response with {len(parsed_data.get('challenges', []))} challenges")
                return parsed_data
                
        except json.JSONDecodeError as e:
            print(f"[SWARM_PARSE] ERROR: Agent {self.id} JSON decode failed: {e}")
        except Exception as e:
            print(f"[SWARM_PARSE] ERROR: Agent {self.id} parsing failed: {e}")
        
        # Enhanced fallback with logging
        print(f"[SWARM_PARSE] Agent {self.id} falling back to default structure due to parse failure")
        fallback_response = {
            'key_points': [response[:200]] if response else ["No response received"],
            'challenges': ["Unable to parse specific challenges from AI response"],
            'recommendations': ["Review AI response format and improve parsing"],
            'priority': 5,
            'complexity': 'medium',
            'confidence': 0.3,  # Lower confidence for fallback
            'raw_response': response,
            'parse_error': True
        }
        
        print(f"[SWARM_PARSE] Agent {self.id} created fallback response with {len(fallback_response['challenges'])} challenges")
        return fallback_response


class RealSwarmIntelligence:
    """Orchestrates multiple real AI agents for emergent intelligence."""
    
    def __init__(self, ai_brain=None, num_agents: int = 7):
        """Initialize swarm with diverse AI-powered agents."""
        self.ai_brain = ai_brain
        self.agents = self._create_real_agents(num_agents)
        self.collective_decisions = []
        self.swarm_performance_metrics = {
            'consensus_rate': 0.0,
            'decision_quality': 0.0,
            'response_time': 0.0
        }
        self.executor = ThreadPoolExecutor(max_workers=num_agents)
        
    def _create_real_agents(self, num_agents: int) -> List[RealSwarmAgent]:
        """Create diverse AI-powered agents with different models and personas."""
        agents = []
        
        # Define real agent configurations
        agent_configs = [
            {
                "role": AgentRole.ARCHITECT,
                "model": "claude-3-opus",
                "expertise": ["system-design", "scalability", "microservices", "cloud-architecture"],
                "persona": "You are a senior system architect with 15+ years of experience. You think in terms of scalability, maintainability, and long-term system evolution. You always consider the bigger picture and how components fit together."
            },
            {
                "role": AgentRole.DEVELOPER,
                "model": "gpt-4",
                "expertise": ["full-stack", "laravel", "react", "api-development", "database-design"],
                "persona": "You are a pragmatic full-stack developer who values clean code, best practices, and rapid delivery. You balance perfectionism with getting things done and always think about developer experience."
            },
            {
                "role": AgentRole.SECURITY,
                "model": "claude-3-opus",
                "expertise": ["cybersecurity", "penetration-testing", "owasp", "encryption", "auth"],
                "persona": "You are a security expert with a hacker's mindset. You think like an attacker to defend better. You're paranoid about security but practical about implementation trade-offs."
            },
            {
                "role": AgentRole.TESTER,
                "model": "gemini-pro",
                "expertise": ["test-automation", "quality-assurance", "tdd", "e2e-testing"],
                "persona": "You are a quality champion who believes that good testing is the foundation of reliable software. You think about edge cases, user scenarios, and what could go wrong."
            },
            {
                "role": AgentRole.STRATEGIST,
                "model": "gpt-4",
                "expertise": ["business-strategy", "product-management", "market-analysis", "user-research"],
                "persona": "You are a strategic thinker who connects technology decisions to business outcomes. You always ask 'why' and focus on delivering value to users and stakeholders."
            },
            {
                "role": AgentRole.PERFORMANCE,
                "model": "claude-3-opus",
                "expertise": ["optimization", "profiling", "caching", "database-tuning", "scalability"],
                "persona": "You are obsessed with performance and efficiency. You measure everything and believe that speed is a feature. You balance optimization with code clarity."
            },
            {
                "role": AgentRole.ORCHESTRATOR,
                "model": "gemini-pro",
                "expertise": ["project-management", "coordination", "risk-assessment", "decision-making"],
                "persona": "You are a master coordinator who sees the big picture and ensures all pieces work together. You synthesize different viewpoints and drive towards consensus."
            }
        ]
        
        # Create agents from configurations
        for i, config in enumerate(agent_configs[:num_agents]):
            agent = RealSwarmAgent(
                id=f"agent_{config['role'].value}_{i}",
                role=config['role'],
                model_name=config['model'],
                expertise_areas=config['expertise'],
                persona=config['persona'],
                ai_brain=self.ai_brain,
                task_history=[]
            )
            agents.append(agent)
            
        return agents
    
    async def process_task_swarm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using the full swarm intelligence."""
        start_time = datetime.now(timezone.utc)
        
        # Phase 1: Individual Analysis - Each agent analyzes independently
        individual_analyses = await self._phase_individual_analysis(task)
        
        # Phase 2: Cross-Pollination - Agents share insights
        refined_analyses = await self._phase_cross_pollination(task, individual_analyses)
        
        # Phase 3: Consensus Building - Synthesize collective intelligence
        consensus = await self._phase_consensus_building(refined_analyses)
        
        # Phase 4: Action Planning - Generate concrete recommendations
        action_plan = await self._phase_action_planning(task, consensus)
        
        # Calculate metrics
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        
        result = {
            'task_id': task.get('id', 'unknown'),
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'individual_analyses': individual_analyses,
            'refined_analyses': refined_analyses,
            'consensus': consensus,
            'action_plan': action_plan,
            'collective_review': self._generate_collective_review(consensus, action_plan)
        }
        
        # Update metrics
        self._update_swarm_metrics(result)
        
        return result
    
    async def _phase_individual_analysis(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Phase 1: Each agent analyzes the task independently."""
        analyses = []
        
        # Create async tasks for parallel analysis
        async_tasks = []
        for agent in self.agents:
            async_tasks.append(agent.analyze_task(task))
        
        # Wait for all analyses to complete
        analyses = await asyncio.gather(*async_tasks)
        
        return analyses
    
    async def _phase_cross_pollination(self, task: Dict[str, Any], 
                                      initial_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 2: Agents refine their analysis based on others' insights."""
        refined_analyses = []
        
        # Each agent gets to see others' analyses and refine their own
        async_tasks = []
        for i, agent in enumerate(self.agents):
            # Get other agents' analyses
            other_analyses = [a for j, a in enumerate(initial_analyses) if j != i]
            async_tasks.append(agent.analyze_task(task, other_analyses))
        
        refined_analyses = await asyncio.gather(*async_tasks)
        
        return refined_analyses
    
    async def _phase_consensus_building(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 3: Build consensus from all analyses."""
        # Extract key themes
        all_key_points = []
        all_challenges = []
        all_recommendations = []
        priorities = []
        
        for analysis in analyses:
            if 'error' not in analysis:
                all_key_points.extend(analysis.get('key_points', []))
                all_challenges.extend(analysis.get('challenges', []))
                all_recommendations.extend(analysis.get('recommendations', []))
                priorities.append(analysis.get('priority', 5))
        
        # Use orchestrator to synthesize if available
        orchestrator = next((a for a in self.agents if a.role == AgentRole.ORCHESTRATOR), None)
        
        if orchestrator and self.ai_brain:
            synthesis_prompt = f"""
            As the swarm orchestrator, synthesize these analyses into a unified consensus:
            
            Key Points from all agents:
            {json.dumps(all_key_points, indent=2)}
            
            Challenges identified:
            {json.dumps(all_challenges, indent=2)}
            
            Recommendations:
            {json.dumps(all_recommendations, indent=2)}
            
            Create a consensus that:
            1. Identifies the most important insights (top 5)
            2. Prioritizes the critical challenges (top 3)
            3. Selects the best recommendations (top 5)
            4. Resolves any conflicting viewpoints
            5. Provides a unified priority score
            
            Format as JSON with keys: key_insights, critical_challenges, 
            top_recommendations, conflicts_resolved, consensus_priority
            """
            
            response = await orchestrator._call_ai_model(synthesis_prompt)
            consensus = orchestrator._parse_ai_response(response)
        else:
            # Fallback: simple aggregation
            consensus = {
                'key_insights': list(set(all_key_points))[:5],
                'critical_challenges': list(set(all_challenges))[:3],
                'top_recommendations': list(set(all_recommendations))[:5],
                'consensus_priority': np.mean(priorities) if priorities else 5
            }
        
        return consensus
    
    async def _phase_action_planning(self, task: Dict[str, Any], 
                                   consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Generate concrete action plan based on consensus."""
        # Use strategist and architect to create action plan
        strategist = next((a for a in self.agents if a.role == AgentRole.STRATEGIST), None)
        architect = next((a for a in self.agents if a.role == AgentRole.ARCHITECT), None)
        
        action_plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_considerations': [],
            'success_metrics': [],
            'risk_mitigation': []
        }
        
        if strategist and self.ai_brain:
            planning_prompt = f"""
            Based on the swarm consensus, create a concrete action plan:
            
            Task: {task.get('title', 'Unknown')}
            Consensus Insights: {json.dumps(consensus.get('key_insights', []))}
            Challenges: {json.dumps(consensus.get('critical_challenges', []))}
            Recommendations: {json.dumps(consensus.get('top_recommendations', []))}
            
            Generate:
            1. Immediate actions (next 24 hours)
            2. Short-term goals (next week)
            3. Long-term considerations
            4. Success metrics
            5. Risk mitigation strategies
            
            Format as JSON with the structure matching the requested sections.
            """
            
            response = await strategist._call_ai_model(planning_prompt)
            plan = strategist._parse_ai_response(response)
            
            # Merge with action_plan structure
            if isinstance(plan, dict):
                action_plan.update(plan)
        
        return action_plan
    
    def _generate_collective_review(self, consensus: Dict[str, Any], 
                                  action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a collective review summary."""
        return {
            'summary': f"Swarm analyzed task with {len(self.agents)} specialized agents",
            'consensus_priority': consensus.get('consensus_priority', 5),
            'key_insights': consensus.get('key_insights', [])[:3],
            'immediate_actions': action_plan.get('immediate_actions', [])[:3],
            'confidence_level': self._calculate_swarm_confidence(consensus),
            'top_suggestions': self._extract_top_suggestions(consensus, action_plan)
        }
    
    def _calculate_swarm_confidence(self, consensus: Dict[str, Any]) -> float:
        """Calculate overall swarm confidence in the analysis."""
        # Simple confidence calculation based on consensus
        factors = []
        
        # Factor 1: Number of insights
        insight_count = len(consensus.get('key_insights', []))
        factors.append(min(insight_count / 5, 1.0))
        
        # Factor 2: Recommendation count
        rec_count = len(consensus.get('top_recommendations', []))
        factors.append(min(rec_count / 5, 1.0))
        
        # Factor 3: Challenge identification
        challenge_count = len(consensus.get('critical_challenges', []))
        factors.append(min(challenge_count / 3, 1.0))
        
        return np.mean(factors) if factors else 0.5
    
    def _extract_top_suggestions(self, consensus: Dict[str, Any], 
                               action_plan: Dict[str, Any]) -> List[str]:
        """Extract top actionable suggestions."""
        suggestions = []
        
        # Add top recommendations
        for rec in consensus.get('top_recommendations', [])[:2]:
            if isinstance(rec, str):
                suggestions.append(rec)
        
        # Add immediate actions
        for action in action_plan.get('immediate_actions', [])[:2]:
            if isinstance(action, str):
                suggestions.append(action)
        
        return suggestions[:3]  # Limit to top 3
    
    def _update_swarm_metrics(self, result: Dict[str, Any]) -> None:
        """Update swarm performance metrics."""
        # Update consensus rate (how well agents agreed)
        analyses = result.get('refined_analyses', [])
        if analyses:
            priorities = [a.get('priority', 5) for a in analyses if 'error' not in a]
            if priorities:
                variance = np.var(priorities)
                self.swarm_performance_metrics['consensus_rate'] = 1 / (1 + variance)
        
        # Update response time
        duration = result.get('duration_seconds', 0)
        self.swarm_performance_metrics['response_time'] = duration
        
        # Update decision quality (based on confidence)
        confidence = result.get('collective_review', {}).get('confidence_level', 0.5)
        self.swarm_performance_metrics['decision_quality'] = confidence
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status and metrics."""
        return {
            'active_agents': len(self.agents),
            'agent_roles': [a.role.value for a in self.agents],
            'performance_metrics': self.swarm_performance_metrics,
            'total_decisions': len(self.collective_decisions)
        }