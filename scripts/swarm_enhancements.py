"""
Swarm Intelligence Enhancement Methods

Additional methods for the enhanced swarm intelligence system.
Contains helper functions for optimization, caching, and intelligence integration.
"""

import hashlib
import json
import time
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
import numpy as np
from collections import defaultdict


class SwarmEnhancements:
    """Helper methods for enhanced swarm intelligence."""
    
    def _generate_task_cache_key(self, task: Dict[str, Any]) -> str:
        """Generate cache key for task similarity matching."""
        # Use task type, title, and key requirements for cache key
        cache_data = {
            'type': task.get('type', ''),
            'title': task.get('title', ''),
            'requirements': sorted(task.get('requirements', [])),
            'description_hash': hashlib.md5(
                task.get('description', '').encode()
            ).hexdigest()[:8]
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()[:16]
    
    def _check_analysis_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if similar task analysis exists in cache."""
        if cache_key in self.analysis_cache:
            cached_entry = self.analysis_cache[cache_key]
            
            # Check if cache entry is still valid (1 hour TTL)
            cache_time = cached_entry.get('timestamp', 0)
            if time.time() - cache_time < 3600:
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return cached_entry['result']
            else:
                # Remove expired entry
                del self.analysis_cache[cache_key]
        
        return None
    
    def _cache_analysis_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result for future use."""
        # Limit cache size
        if len(self.analysis_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.analysis_cache.keys(), 
                           key=lambda k: self.analysis_cache[k]['timestamp'])
            del self.analysis_cache[oldest_key]
        
        self.analysis_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _select_optimal_agents(self, task: Dict[str, Any]):
        """Select optimal agents based on workload and task type."""
        # Sort agents by current workload
        available_agents = sorted(
            self.agents, 
            key=lambda a: self.agent_workload[a.id]
        )
        
        # Select agents based on task type and role relevance
        task_type = task.get('type', 'general')
        selected_agents = []
        
        # Always include orchestrator if available
        orchestrator = next((a for a in available_agents if a.role.value == 'orchestrator'), None)
        if orchestrator:
            selected_agents.append(orchestrator)
        
        # Select specialized agents based on task type
        role_priority = self._get_role_priority_for_task(task_type)
        
        for role in role_priority:
            agent = next((a for a in available_agents if a.role.value == role), None)
            if agent and agent not in selected_agents:
                selected_agents.append(agent)
                
                # Limit to avoid overwhelming the system
                if len(selected_agents) >= 5:
                    break
        
        # Fill remaining slots with least loaded agents
        while len(selected_agents) < min(len(self.agents), 7):
            for agent in available_agents:
                if agent not in selected_agents:
                    selected_agents.append(agent)
                    break
            else:
                break
        
        return selected_agents
    
    def _get_role_priority_for_task(self, task_type: str):
        """Get role priority order based on task type."""
        role_priorities = {
            'bug_fix': ['developer', 'tester', 'security', 'reviewer'],
            'feature_development': ['architect', 'developer', 'tester', 'performance'],
            'security_audit': ['security', 'reviewer', 'architect', 'tester'],
            'performance_optimization': ['performance', 'architect', 'developer'],
            'code_review': ['reviewer', 'security', 'architect', 'developer'],
            'research': ['researcher', 'strategist', 'architect'],
            'strategic_planning': ['strategist', 'architect', 'researcher'],
            'general': ['architect', 'developer', 'tester', 'security', 'reviewer']
        }
        
        return role_priorities.get(task_type, role_priorities['general'])
    
    def _calculate_agent_timeout(self, agent) -> float:
        """Calculate timeout for agent based on historical performance."""
        # Base timeout
        base_timeout = 30.0
        
        # Adjust based on agent role complexity
        role_multipliers = {
            'orchestrator': 1.5,
            'architect': 1.3,
            'strategist': 1.2,
            'researcher': 1.1,
            'developer': 1.0,
            'tester': 0.9,
            'reviewer': 0.8,
            'security': 1.0,
            'performance': 0.9
        }
        
        multiplier = role_multipliers.get(agent.role.value, 1.0)
        
        # Adjust based on current workload
        workload_factor = 1.0 + (self.agent_workload[agent.id] * 0.1)
        
        return base_timeout * multiplier * workload_factor
    
    def _filter_high_quality_insights(self, analyses):
        """Filter analyses to keep only high-quality insights."""
        high_quality = []
        
        for analysis in analyses:
            if 'error' in analysis:
                continue
            
            # Quality criteria
            confidence = analysis.get('confidence', 0.5)
            has_recommendations = len(analysis.get('recommendations', [])) > 0
            has_challenges = len(analysis.get('challenges', [])) > 0
            
            # Quality score
            quality_score = confidence
            if has_recommendations:
                quality_score += 0.2
            if has_challenges:
                quality_score += 0.1
            
            if quality_score >= 0.6:
                analysis['quality_score'] = quality_score
                high_quality.append(analysis)
        
        # Sort by quality and return top insights
        high_quality.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        return high_quality[:10]  # Top 10 insights
    
    def _get_relevant_insights_for_agent(self, agent, insights):
        """Get insights relevant to specific agent role."""
        relevant = []
        agent_role = agent.role.value
        
        # Role-based insight filtering
        role_interests = {
            'architect': ['scalability', 'design', 'patterns', 'architecture'],
            'developer': ['implementation', 'coding', 'frameworks', 'libraries'],
            'tester': ['testing', 'quality', 'bugs', 'validation'],
            'security': ['security', 'vulnerabilities', 'encryption', 'auth'],
            'performance': ['optimization', 'speed', 'efficiency', 'caching'],
            'researcher': ['trends', 'innovation', 'market', 'future'],
            'strategist': ['business', 'strategy', 'goals', 'planning'],
            'reviewer': ['quality', 'standards', 'best practices', 'review'],
            'orchestrator': []  # Orchestrator gets all insights
        }
        
        interests = role_interests.get(agent_role, [])
        
        for insight in insights:
            if agent_role == 'orchestrator':
                # Orchestrator gets all insights
                relevant.append(insight)
            else:
                # Check if insight is relevant to agent's interests
                insight_text = json.dumps(insight).lower()
                if any(interest in insight_text for interest in interests):
                    relevant.append(insight)
        
        # Limit to top 5 relevant insights
        return relevant[:5]
    
    def _extract_weighted_insights(self, analyses):
        """Extract insights with agent-based weighting."""
        weighted_insights = {
            'key_points': defaultdict(list),
            'challenges': defaultdict(list),
            'recommendations': defaultdict(list),
            'priorities': []
        }
        
        # Agent role weights
        role_weights = {
            'orchestrator': 1.5,
            'architect': 1.3,
            'strategist': 1.2,
            'security': 1.1,
            'developer': 1.0,
            'tester': 0.9,
            'reviewer': 0.9,
            'researcher': 0.8,
            'performance': 0.8
        }
        
        for analysis in analyses:
            if 'error' in analysis:
                continue
            
            agent_role = analysis.get('agent_role', 'unknown')
            weight = role_weights.get(agent_role, 1.0)
            confidence = analysis.get('confidence', 0.5)
            
            # Combined weight
            total_weight = weight * confidence
            
            # Add weighted insights
            for point in analysis.get('key_points', []):
                weighted_insights['key_points'][point].append(total_weight)
            
            for challenge in analysis.get('challenges', []):
                weighted_insights['challenges'][challenge].append(total_weight)
            
            for rec in analysis.get('recommendations', []):
                weighted_insights['recommendations'][rec].append(total_weight)
            
            # Weight priority
            priority = analysis.get('priority', 5)
            weighted_insights['priorities'].append(priority * total_weight)
        
        # Calculate final weighted scores
        final_insights = {}
        
        for category in ['key_points', 'challenges', 'recommendations']:
            items = []
            for item, weights in weighted_insights[category].items():
                avg_weight = np.mean(weights)
                items.append({'item': item, 'weight': avg_weight, 'count': len(weights)})
            
            # Sort by weight and select top items
            items.sort(key=lambda x: x['weight'], reverse=True)
            final_insights[category] = [item['item'] for item in items[:10]]
        
        # Calculate weighted priority
        if weighted_insights['priorities']:
            final_insights['consensus_priority'] = np.mean(weighted_insights['priorities'])
        else:
            final_insights['consensus_priority'] = 5.0
        
        return final_insights
    
    def _detect_analysis_conflicts(self, analyses):
        """Detect conflicts between agent analyses."""
        conflicts = []
        
        # Check priority conflicts
        priorities = []
        for analysis in analyses:
            if 'error' not in analysis:
                priority = analysis.get('priority', 5)
                agent_role = analysis.get('agent_role', 'unknown')
                priorities.append({'priority': priority, 'agent': agent_role})
        
        if len(priorities) > 1:
            priority_values = [p['priority'] for p in priorities]
            if max(priority_values) - min(priority_values) > 5:
                conflicts.append({
                    'type': 'priority_conflict',
                    'description': f"High priority variance: {min(priority_values)}-{max(priority_values)}",
                    'agents_involved': [p['agent'] for p in priorities]
                })
        
        # Check recommendation conflicts
        all_recommendations = []
        for analysis in analyses:
            if 'error' not in analysis:
                for rec in analysis.get('recommendations', []):
                    all_recommendations.append({
                        'recommendation': rec,
                        'agent': analysis.get('agent_role', 'unknown')
                    })
        
        # Simple conflict detection: opposing recommendations
        conflict_keywords = [
            ('implement', 'avoid'),
            ('increase', 'decrease'),
            ('add', 'remove'),
            ('upgrade', 'downgrade')
        ]
        
        for rec1 in all_recommendations:
            for rec2 in all_recommendations:
                if rec1['agent'] != rec2['agent']:
                    for conflict_pair in conflict_keywords:
                        if (conflict_pair[0] in rec1['recommendation'].lower() and 
                            conflict_pair[1] in rec2['recommendation'].lower()):
                            conflicts.append({
                                'type': 'recommendation_conflict',
                                'description': f"Conflicting recommendations between {rec1['agent']} and {rec2['agent']}",
                                'recommendations': [rec1['recommendation'], rec2['recommendation']]
                            })
        
        return conflicts
    
    async def _resolve_conflicts(self, conflicts):
        """Resolve detected conflicts using AI analysis."""
        if not conflicts:
            return []
        
        resolved = []
        
        for conflict in conflicts:
            resolution = {
                'original_conflict': conflict,
                'resolution_strategy': 'default',
                'resolved_recommendation': None
            }
            
            if conflict['type'] == 'priority_conflict':
                # Use median priority as resolution
                resolution['resolution_strategy'] = 'median_priority'
                resolution['resolved_recommendation'] = 'Use median priority from all agents'
            
            elif conflict['type'] == 'recommendation_conflict':
                # Try to synthesize recommendations
                rec1, rec2 = conflict['recommendations']
                resolution['resolution_strategy'] = 'synthesis'
                resolution['resolved_recommendation'] = f"Consider both approaches: {rec1[:50]}... and {rec2[:50]}..."
            
            resolved.append(resolution)
        
        return resolved
    
    def _get_agent_performance_context(self) -> Dict[str, Any]:
        """Get agent performance context for consensus building."""
        context = {}
        
        for agent in self.agents:
            agent_id = agent.id
            context[agent_id] = {
                'role': agent.role.value,
                'current_workload': self.agent_workload[agent_id],
                'performance_score': agent.performance_score,
                'task_count': len(agent.task_history)
            }
        
        return context
    
    def _calculate_agent_weights(self, analyses):
        """Calculate agent weights based on analysis quality."""
        weights = {}
        
        for analysis in analyses:
            if 'error' not in analysis:
                agent_id = analysis.get('agent_id', 'unknown')
                confidence = analysis.get('confidence', 0.5)
                
                # Quality indicators
                has_recommendations = len(analysis.get('recommendations', [])) > 0
                has_challenges = len(analysis.get('challenges', [])) > 0
                has_key_points = len(analysis.get('key_points', [])) > 0
                
                # Calculate weight
                weight = confidence
                if has_recommendations:
                    weight += 0.2
                if has_challenges:
                    weight += 0.1
                if has_key_points:
                    weight += 0.1
                
                weights[agent_id] = min(weight, 1.0)
        
        return weights
    
    def _calculate_consensus_quality(self, analyses):
        """Calculate overall consensus quality score."""
        if not analyses:
            return 0.0
        
        valid_analyses = [a for a in analyses if 'error' not in a]
        if not valid_analyses:
            return 0.0
        
        # Quality factors
        avg_confidence = np.mean([a.get('confidence', 0.5) for a in valid_analyses])
        
        # Consistency in priorities
        priorities = [a.get('priority', 5) for a in valid_analyses]
        priority_variance = np.var(priorities) if len(priorities) > 1 else 0
        priority_consistency = 1.0 / (1.0 + priority_variance)
        
        # Completeness
        completeness = np.mean([
            1.0 if (len(a.get('recommendations', [])) > 0 and 
                   len(a.get('challenges', [])) > 0 and 
                   len(a.get('key_points', [])) > 0) else 0.5
            for a in valid_analyses
        ])
        
        # Combined quality score
        quality = (avg_confidence * 0.4 + 
                  priority_consistency * 0.3 + 
                  completeness * 0.3)
        
        return min(quality, 1.0)
    
    def _create_enhanced_fallback_consensus(self, weighted_insights, analyses):
        """Create enhanced fallback consensus when orchestrator is unavailable."""
        return {
            'key_insights': weighted_insights.get('key_points', [])[:5],
            'critical_challenges': weighted_insights.get('challenges', [])[:3],
            'strategic_recommendations': weighted_insights.get('recommendations', [])[:5],
            'consensus_priority': weighted_insights.get('consensus_priority', 5.0),
            'confidence_level': self._calculate_consensus_quality(analyses),
            'agent_weights': self._calculate_agent_weights(analyses),
            'consensus_method': 'weighted_aggregation_fallback'
        }
    
    async def _phase_strategic_action_planning(self, task, consensus):
        """Enhanced strategic action planning with context awareness."""
        # Get strategist and architect for planning
        strategist = next((a for a in self.agents if a.role.value == 'strategist'), None)
        architect = next((a for a in self.agents if a.role.value == 'architect'), None)
        
        action_plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_considerations': [],
            'success_metrics': [],
            'risk_mitigation': [],
            'resource_requirements': [],
            'timeline': {}
        }
        
        if strategist and self.ai_brain:
            # Enhanced planning prompt with context
            planning_prompt = f"""
            Create a comprehensive strategic action plan based on swarm consensus:
            
            Task Context: {task.get('title', 'Unknown')}
            Task Type: {task.get('type', 'general')}
            
            Consensus Analysis:
            Key Insights: {json.dumps(consensus.get('key_insights', []))}
            Critical Challenges: {json.dumps(consensus.get('critical_challenges', []))}
            Strategic Recommendations: {json.dumps(consensus.get('strategic_recommendations', []))}
            Priority Level: {consensus.get('consensus_priority', 5)}
            Confidence: {consensus.get('confidence_level', 0.5)}
            
            Performance Context:
            Current System Performance: {json.dumps(self.swarm_performance_metrics)}
            Recent Performance History: {len(self.performance_history)} recent tasks
            
            Generate a strategic action plan with:
            1. Immediate actions (next 24-48 hours)
            2. Short-term goals (next week)
            3. Long-term considerations (next month)
            4. Measurable success metrics
            5. Risk mitigation strategies
            6. Resource requirements
            7. Timeline with milestones
            
            Format as JSON with detailed, actionable items.
            """
            
            response = await strategist._call_ai_model(planning_prompt)
            plan = strategist._parse_ai_response(response)
            
            # Merge with action plan structure
            if isinstance(plan, dict):
                action_plan.update(plan)
        
        # Add system-generated enhancements
        action_plan['planning_metadata'] = {
            'planning_agent': strategist.id if strategist else None,
            'planning_timestamp': datetime.now(timezone.utc).isoformat(),
            'consensus_quality': consensus.get('confidence_level', 0.5),
            'swarm_performance_context': dict(self.swarm_performance_metrics)
        }
        
        return action_plan
    
    def _calculate_task_performance(self, analyses, duration):
        """Calculate performance metrics for the task."""
        valid_analyses = [a for a in analyses if 'error' not in a]
        
        return {
            'analysis_success_rate': len(valid_analyses) / len(analyses) if analyses else 0,
            'average_confidence': np.mean([a.get('confidence', 0.5) for a in valid_analyses]) if valid_analyses else 0,
            'processing_time': duration,
            'agents_utilized': len(analyses),
            'insights_generated': sum(len(a.get('key_points', [])) for a in valid_analyses),
            'recommendations_count': sum(len(a.get('recommendations', [])) for a in valid_analyses),
            'challenges_identified': sum(len(a.get('challenges', [])) for a in valid_analyses)
        }
    
    def _update_enhanced_swarm_metrics(self, result):
        """Update enhanced swarm performance metrics."""
        # Update existing metrics
        self._update_swarm_metrics(result)
        
        # Calculate throughput (tasks per hour)
        current_time = time.time()
        recent_tasks = [
            task for task in self.performance_history 
            if current_time - task.get('timestamp', datetime.now()).timestamp() < 3600
        ]
        self.swarm_performance_metrics['throughput'] = len(recent_tasks)
        
        # Update cache hit rate
        total_requests = len(self.performance_history) + 1
        cache_hits = sum(1 for task in self.performance_history if task.get('cache_used', False))
        self.swarm_performance_metrics['cache_hit_rate'] = cache_hits / total_requests if total_requests > 0 else 0
    
    async def set_intelligence_hub(self, hub):
        """Set intelligence hub for event emission."""
        self.intelligence_hub = hub
        self.logger.info("Intelligence hub connected to swarm intelligence")