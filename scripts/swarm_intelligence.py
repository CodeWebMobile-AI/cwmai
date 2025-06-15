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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from datetime import datetime, timezone
import logging
import time
from collections import defaultdict, deque
from scripts.mcp_redis_integration import MCPRedisIntegration


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
    PLANNER = "planner"            # Strategic planning and project management
    CODER = "coder"                # Specialized code generation
    QA_ENGINEER = "qa_engineer"    # Quality assurance and testing
    SECURITY_ANALYST = "security_analyst"  # Security vulnerability analysis
    DOCUMENTER = "documenter"      # Documentation and technical writing


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
            'response_time': 0.0,
            'throughput': 0.0,
            'cache_hit_rate': 0.0
        }
        self.executor = ThreadPoolExecutor(max_workers=num_agents)
        
        # Enhanced processing features
        self.task_queue = deque()
        self.processing_batch_size = 5
        self.agent_workload = defaultdict(int)
        self.analysis_cache = {}  # Cache for similar task analyses
        self.performance_history = deque(maxlen=100)
        
        # Intelligence integration
        self.intelligence_hub = None
        self.logger = logging.getLogger(f"{__name__}.RealSwarmIntelligence")
        
        # MCP-Redis integration
        self.mcp_redis: Optional[MCPRedisIntegration] = None
        self._use_mcp = os.getenv("USE_MCP_REDIS", "false").lower() == "true"
        
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
    
    # Include all enhancement methods
    from scripts.swarm_enhancements import SwarmEnhancements
    
    # Mix in enhancement methods
    _generate_task_cache_key = SwarmEnhancements._generate_task_cache_key
    _check_analysis_cache = SwarmEnhancements._check_analysis_cache
    _cache_analysis_result = SwarmEnhancements._cache_analysis_result
    _select_optimal_agents = SwarmEnhancements._select_optimal_agents
    _get_role_priority_for_task = SwarmEnhancements._get_role_priority_for_task
    _calculate_agent_timeout = SwarmEnhancements._calculate_agent_timeout
    _filter_high_quality_insights = SwarmEnhancements._filter_high_quality_insights
    _get_relevant_insights_for_agent = SwarmEnhancements._get_relevant_insights_for_agent
    _extract_weighted_insights = SwarmEnhancements._extract_weighted_insights
    _detect_analysis_conflicts = SwarmEnhancements._detect_analysis_conflicts
    _resolve_conflicts = SwarmEnhancements._resolve_conflicts
    _get_agent_performance_context = SwarmEnhancements._get_agent_performance_context
    _calculate_agent_weights = SwarmEnhancements._calculate_agent_weights
    _calculate_consensus_quality = SwarmEnhancements._calculate_consensus_quality
    _create_enhanced_fallback_consensus = SwarmEnhancements._create_enhanced_fallback_consensus
    _phase_strategic_action_planning = SwarmEnhancements._phase_strategic_action_planning
    _calculate_task_performance = SwarmEnhancements._calculate_task_performance
    _update_enhanced_swarm_metrics = SwarmEnhancements._update_enhanced_swarm_metrics
    set_intelligence_hub = SwarmEnhancements.set_intelligence_hub
    
    async def process_task_swarm(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using the full swarm intelligence with enhanced optimization."""
        start_time = datetime.now(timezone.utc)
        
        # Check cache for similar tasks first
        cache_key = self._generate_task_cache_key(task)
        cached_result = self._check_analysis_cache(cache_key)
        if cached_result:
            self.swarm_performance_metrics['cache_hit_rate'] += 1
            self.logger.debug(f"Cache hit for task: {task.get('title', 'unknown')}")
            return cached_result
        
        # Emit intelligence event
        if self.intelligence_hub:
            await self.intelligence_hub.emit_event(
                event_type="swarm_analysis",
                source_component="swarm_intelligence",
                data={
                    "task_id": task.get('id', 'unknown'),
                    "task_type": task.get('type', 'unknown'),
                    "phase": "start"
                }
            )
        
        # Phase 1: Enhanced Individual Analysis - Parallel processing with load balancing
        individual_analyses = await self._phase_enhanced_individual_analysis(task)
        
        # Phase 2: Intelligent Cross-Pollination - Optimized insight sharing
        refined_analyses = await self._phase_intelligent_cross_pollination(task, individual_analyses)
        
        # Phase 3: Advanced Consensus Building - Weighted consensus with conflict resolution
        consensus = await self._phase_advanced_consensus_building(refined_analyses)
        
        # Phase 4: Strategic Action Planning - Context-aware planning
        action_plan = await self._phase_strategic_action_planning(task, consensus)
        
        # Calculate metrics and performance
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
            'collective_review': self._generate_collective_review(consensus, action_plan),
            'performance_metrics': self._calculate_task_performance(individual_analyses, duration),
            'cache_used': cached_result is not None
        }
        
        # Cache the result for future similar tasks
        self._cache_analysis_result(cache_key, result)
        
        # Update swarm metrics and history
        self._update_enhanced_swarm_metrics(result)
        self.performance_history.append({
            'timestamp': start_time,
            'duration': duration,
            'task_type': task.get('type', 'unknown'),
            'agents_used': len(individual_analyses),
            'consensus_quality': consensus.get('consensus_priority', 5)
        })
        
        # Emit completion event
        if self.intelligence_hub:
            await self.intelligence_hub.emit_event(
                event_type="swarm_analysis",
                source_component="swarm_intelligence", 
                data={
                    "task_id": task.get('id', 'unknown'),
                    "phase": "complete",
                    "duration": duration,
                    "consensus_priority": consensus.get('consensus_priority', 5)
                }
            )
        
        return result
    
    async def _phase_enhanced_individual_analysis(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Phase 1: Enhanced individual analysis with load balancing and optimization."""
        start_time = time.time()
        
        # Select agents based on workload balancing and task type
        selected_agents = self._select_optimal_agents(task)
        
        # Create async tasks for parallel analysis with timeout
        async_tasks = []
        for agent in selected_agents:
            # Update agent workload
            self.agent_workload[agent.id] += 1
            
            # Create task with timeout based on agent performance
            timeout = self._calculate_agent_timeout(agent)
            async_tasks.append(
                asyncio.wait_for(agent.analyze_task(task), timeout=timeout)
            )
        
        # Execute with exception handling
        analyses = []
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            agent = selected_agents[i]
            # Decrease workload after completion
            self.agent_workload[agent.id] = max(0, self.agent_workload[agent.id] - 1)
            
            if isinstance(result, Exception):
                self.logger.warning(f"Agent {agent.id} analysis failed: {result}")
                # Create fallback analysis
                fallback = {
                    'agent_id': agent.id,
                    'agent_role': agent.role.value,
                    'error': str(result),
                    'key_points': [],
                    'recommendations': [],
                    'priority': 5,
                    'complexity': 'unknown',
                    'confidence': 0.1
                }
                analyses.append(fallback)
            else:
                analyses.append(result)
        
        # Update performance metrics
        duration = time.time() - start_time
        self.swarm_performance_metrics['response_time'] = duration
        
        self.logger.info(f"Individual analysis completed: {len(analyses)} agents, {duration:.2f}s")
        return analyses
    
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
    
    async def _phase_intelligent_cross_pollination(self, task: Dict[str, Any], 
                                                 initial_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 2: Intelligent cross-pollination with selective insight sharing."""
        refined_analyses = []
        
        # Filter and rank insights for better cross-pollination
        high_quality_insights = self._filter_high_quality_insights(initial_analyses)
        
        # Each agent gets curated insights based on their role
        async_tasks = []
        for i, agent in enumerate(self.agents):
            if i < len(initial_analyses):
                # Get relevant insights for this agent's role
                relevant_insights = self._get_relevant_insights_for_agent(agent, high_quality_insights)
                async_tasks.append(agent.analyze_task(task, relevant_insights))
        
        refined_analyses = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_analyses = []
        for i, result in enumerate(refined_analyses):
            if isinstance(result, Exception):
                # Fall back to initial analysis
                if i < len(initial_analyses):
                    processed_analyses.append(initial_analyses[i])
            else:
                processed_analyses.append(result)
        
        return processed_analyses
    
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
    
    async def _phase_advanced_consensus_building(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 3: Advanced consensus building with weighted analysis and conflict resolution."""
        # Extract themes with agent weighting based on confidence and role
        weighted_insights = self._extract_weighted_insights(analyses)
        
        # Detect and resolve conflicts
        conflicts = self._detect_analysis_conflicts(analyses)
        resolved_conflicts = await self._resolve_conflicts(conflicts)
        
        # Use orchestrator with enhanced synthesis
        orchestrator = next((a for a in self.agents if a.role == AgentRole.ORCHESTRATOR), None)
        
        if orchestrator and self.ai_brain:
            synthesis_prompt = f"""
            As the swarm orchestrator, create an advanced consensus from weighted agent analyses:
            
            Weighted Insights:
            {json.dumps(weighted_insights, indent=2)}
            
            Resolved Conflicts:
            {json.dumps(resolved_conflicts, indent=2)}
            
            Agent Performance Context:
            {json.dumps(self._get_agent_performance_context(), indent=2)}
            
            Create a sophisticated consensus that:
            1. Prioritizes insights by agent confidence and expertise
            2. Integrates conflict resolutions
            3. Provides strategic recommendations
            4. Includes risk assessment
            5. Estimates implementation complexity
            6. Suggests success metrics
            
            Format as JSON with keys: key_insights, critical_challenges, 
            strategic_recommendations, risk_assessment, implementation_plan,
            success_metrics, consensus_priority, confidence_level
            """
            
            response = await orchestrator._call_ai_model(synthesis_prompt)
            consensus = orchestrator._parse_ai_response(response)
            
            # Enhance consensus with metadata
            consensus['agent_weights'] = self._calculate_agent_weights(analyses)
            consensus['consensus_quality'] = self._calculate_consensus_quality(analyses)
        else:
            # Enhanced fallback with weighted aggregation
            consensus = self._create_enhanced_fallback_consensus(weighted_insights, analyses)
        
        return consensus
    
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
    
    async def initialize(self):
        """Initialize swarm components including MCP-Redis."""
        if self._use_mcp:
            try:
                self.mcp_redis = MCPRedisIntegration()
                await self.mcp_redis.initialize()
                self.logger.info("MCP-Redis integration enabled for swarm intelligence")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MCP-Redis: {e}")
                self._use_mcp = False
    
    # MCP-Redis Enhanced Methods
    async def analyze_swarm_dynamics(self) -> Dict[str, Any]:
        """Analyze swarm dynamics and agent interactions using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            # Gather agent performance data
            agent_data = []
            for agent in self.agents:
                agent_data.append({
                    'id': agent.id,
                    'role': agent.role.value,
                    'model': agent.model_name,
                    'expertise': agent.expertise_areas,
                    'performance_score': agent.performance_score,
                    'task_count': len(agent.task_history)
                })
            
            analysis = await self.mcp_redis.execute(f"""
                Analyze swarm dynamics with {len(self.agents)} agents:
                
                Agent profiles:
                {json.dumps(agent_data, indent=2)}
                
                Performance metrics:
                - Consensus rate: {self.swarm_performance_metrics['consensus_rate']:.2%}
                - Decision quality: {self.swarm_performance_metrics['decision_quality']:.2%}
                - Average response time: {self.swarm_performance_metrics['response_time']:.2f}s
                - Cache hit rate: {self.swarm_performance_metrics['cache_hit_rate']:.2%}
                
                Analyze:
                - Agent collaboration effectiveness
                - Role coverage and gaps
                - Model diversity benefits
                - Consensus patterns
                - Performance bottlenecks
                - Optimal agent composition
                - Expertise utilization
                
                Provide insights on improving swarm performance.
            """)
            
            return analysis if isinstance(analysis, dict) else {"analysis": analysis}
            
        except Exception as e:
            self.logger.error(f"Error analyzing swarm dynamics: {e}")
            return {"error": str(e)}
    
    async def optimize_agent_composition(self, task_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent composition for specific task types using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            optimization = await self.mcp_redis.execute(f"""
                Optimize agent composition for task profile:
                
                Task characteristics:
                {json.dumps(task_profile, indent=2)}
                
                Current agents:
                - {len(self.agents)} total agents
                - Roles: {[a.role.value for a in self.agents]}
                - Models: {[a.model_name for a in self.agents]}
                
                Performance history:
                - Recent consensus rates: {self.swarm_performance_metrics['consensus_rate']:.2%}
                - Decision quality: {self.swarm_performance_metrics['decision_quality']:.2%}
                
                Recommend:
                - Optimal number of agents
                - Best role distribution
                - Model selection strategy
                - Expertise requirements
                - Agent addition/removal suggestions
                - Specialization adjustments
                
                Consider task complexity, time constraints, and quality requirements.
            """)
            
            return optimization if isinstance(optimization, dict) else {"optimization": optimization}
            
        except Exception as e:
            self.logger.error(f"Error optimizing agent composition: {e}")
            return {"error": str(e)}
    
    async def predict_consensus_quality(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Predict consensus quality before processing using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"confidence": 0.5, "method": "default"}
        
        try:
            prediction = await self.mcp_redis.execute(f"""
                Predict consensus quality for task:
                
                Task: {json.dumps(task, indent=2)}
                
                Available agents:
                - Roles: {[a.role.value for a in self.agents]}
                - Expertise coverage: {list(set(sum([a.expertise_areas for a in self.agents], [])))}
                
                Historical performance:
                - Average consensus rate: {self.swarm_performance_metrics['consensus_rate']:.2%}
                - Task type success rates: analyze from history
                
                Predict:
                - Expected consensus quality (0-1)
                - Confidence in prediction
                - Potential challenges
                - Key success factors
                - Recommended pre-processing
                
                Base prediction on task complexity and agent capabilities.
            """)
            
            return prediction if isinstance(prediction, dict) else {"prediction": prediction}
            
        except Exception as e:
            self.logger.error(f"Error predicting consensus quality: {e}")
            return {"confidence": 0.5, "error": str(e)}
    
    async def analyze_agent_conflicts(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deep analysis of agent conflicts using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return self._detect_analysis_conflicts(analyses)
        
        try:
            # Prepare conflict data
            conflict_data = []
            for i, analysis1 in enumerate(analyses):
                for j, analysis2 in enumerate(analyses[i+1:], i+1):
                    if analysis1.get('priority', 5) != analysis2.get('priority', 5):
                        conflict_data.append({
                            'agents': [analysis1['agent_role'], analysis2['agent_role']],
                            'priority_diff': abs(analysis1.get('priority', 5) - analysis2.get('priority', 5)),
                            'recommendations': {
                                analysis1['agent_role']: analysis1.get('recommendations', []),
                                analysis2['agent_role']: analysis2.get('recommendations', [])
                            }
                        })
            
            conflict_analysis = await self.mcp_redis.execute(f"""
                Analyze conflicts between agent analyses:
                
                Identified conflicts:
                {json.dumps(conflict_data, indent=2)}
                
                Agent analyses summary:
                {json.dumps([{
                    'agent': a['agent_role'],
                    'priority': a.get('priority', 5),
                    'key_points': a.get('key_points', [])[:3]
                } for a in analyses], indent=2)}
                
                Analyze:
                - Root causes of disagreements
                - Valid perspectives from each side
                - Synthesis opportunities
                - Resolution strategies
                - Learning opportunities
                - Future conflict prevention
                
                Provide balanced conflict resolution.
            """)
            
            return conflict_analysis if isinstance(conflict_analysis, dict) else {"analysis": conflict_analysis}
            
        except Exception as e:
            self.logger.error(f"Error analyzing conflicts: {e}")
            return {"error": str(e)}
    
    async def generate_swarm_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive swarm intelligence insights using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            # Gather comprehensive swarm data
            performance_summary = {
                'total_tasks': len(self.collective_decisions),
                'avg_agents_per_task': len(self.agents),
                'consensus_rate': self.swarm_performance_metrics['consensus_rate'],
                'decision_quality': self.swarm_performance_metrics['decision_quality'],
                'avg_response_time': self.swarm_performance_metrics['response_time'],
                'cache_efficiency': self.swarm_performance_metrics['cache_hit_rate']
            }
            
            report = await self.mcp_redis.execute(f"""
                Generate swarm intelligence insights report:
                
                Swarm composition:
                - {len(self.agents)} specialized agents
                - Roles: {[a.role.value for a in self.agents]}
                - Models: {list(set(a.model_name for a in self.agents))}
                
                Performance summary:
                {json.dumps(performance_summary, indent=2)}
                
                Agent utilization:
                {json.dumps(dict(self.agent_workload), indent=2)}
                
                Generate insights on:
                - Swarm effectiveness trends
                - Agent collaboration patterns
                - Decision quality factors
                - Optimization opportunities
                - Scaling recommendations
                - Cost-benefit analysis
                - Future improvements
                
                Format as executive summary with actionable recommendations.
            """)
            
            return report if isinstance(report, dict) else {"report": report}
            
        except Exception as e:
            self.logger.error(f"Error generating insights report: {e}")
            return {"error": str(e)}
    
    async def intelligent_agent_selection(self, task: Dict[str, Any]) -> List[RealSwarmAgent]:
        """Select optimal agents for a task using MCP-Redis intelligence."""
        if not self._use_mcp or not self.mcp_redis:
            return self._select_optimal_agents(task)
        
        try:
            selection = await self.mcp_redis.execute(f"""
                Select optimal agents for task:
                
                Task details:
                - Type: {task.get('type', 'unknown')}
                - Title: {task.get('title', 'untitled')}
                - Requirements: {task.get('requirements', [])}
                - Complexity: analyze from description
                
                Available agents:
                {json.dumps([{
                    'id': a.id,
                    'role': a.role.value,
                    'expertise': a.expertise_areas,
                    'performance': a.performance_score,
                    'workload': self.agent_workload.get(a.id, 0)
                } for a in self.agents], indent=2)}
                
                Select agents based on:
                - Expertise match
                - Historical performance
                - Current workload
                - Synergy between agents
                - Task requirements
                
                Return: list of agent IDs in priority order
            """)
            
            # Extract agent IDs and return corresponding agents
            if isinstance(selection, list):
                selected_agents = []
                for agent_id in selection:
                    agent = next((a for a in self.agents if a.id == agent_id), None)
                    if agent:
                        selected_agents.append(agent)
                return selected_agents if selected_agents else self._select_optimal_agents(task)
            else:
                return self._select_optimal_agents(task)
                
        except Exception as e:
            self.logger.error(f"Error in intelligent agent selection: {e}")
            return self._select_optimal_agents(task)
    
    async def learn_from_decisions(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from swarm decisions to improve future performance using MCP-Redis."""
        if not self._use_mcp or not self.mcp_redis:
            return {"message": "MCP-Redis not available"}
        
        try:
            learning = await self.mcp_redis.execute(f"""
                Learn from swarm decision outcome:
                
                Decision summary:
                - Task type: {result.get('task_id', 'unknown')}
                - Agents involved: {len(result.get('individual_analyses', []))}
                - Consensus priority: {result.get('consensus', {}).get('consensus_priority', 5)}
                - Decision quality: {result.get('collective_review', {}).get('confidence_level', 0)}
                - Processing time: {result.get('duration_seconds', 0):.2f}s
                
                Performance metrics:
                {json.dumps(result.get('performance_metrics', {}), indent=2)}
                
                Extract learnings:
                - What worked well
                - What could improve
                - Agent performance insights
                - Consensus patterns
                - Optimization opportunities
                - Parameter adjustments
                - Training recommendations
                
                Provide specific improvements for next iteration.
            """)
            
            # Store learnings for future reference
            if isinstance(learning, dict) and 'improvements' in learning:
                self.collective_decisions.append({
                    'timestamp': datetime.now(timezone.utc),
                    'result': result,
                    'learnings': learning
                })
            
            return learning if isinstance(learning, dict) else {"learning": learning}
            
        except Exception as e:
            self.logger.error(f"Error learning from decisions: {e}")
            return {"error": str(e)}