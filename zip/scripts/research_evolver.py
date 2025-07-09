"""
Research Evolver

Meta-learning system that evolves research algorithms based on their effectiveness.
Learns which research approaches work best for different types of problems.
"""

import os
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import hashlib


class ResearchEvolver:
    """Evolves research strategies based on effectiveness."""
    
    def __init__(self, ai_brain):
        """Initialize research evolver.
        
        Args:
            ai_brain: AI brain for meta-learning and context gathering
        """
        self.ai_brain = ai_brain
        self.research_history = []
        self.strategy_pool = self._initialize_strategies()
        self.effectiveness_scores = defaultdict(lambda: {'score': 0.5, 'uses': 0})
        self.evolved_strategies = []
        self.meta_insights = []
        
    def _initialize_strategies(self) -> List[Dict[str, Any]]:
        """Initialize base research strategies.
        
        Returns:
            List of research strategies
        """
        return [
            {
                'id': 'broad_survey',
                'name': 'Broad Survey',
                'approach': 'Wide exploration of domain',
                'prompts': [
                    'Provide comprehensive overview of {topic}',
                    'What are all approaches to {topic}',
                    'Survey the landscape of {topic}'
                ],
                'best_for': ['new_domains', 'exploration'],
                'parameters': {'breadth': 0.9, 'depth': 0.3}
            },
            {
                'id': 'deep_dive',
                'name': 'Deep Dive',
                'approach': 'Detailed analysis of specific aspect',
                'prompts': [
                    'Explain in detail how {topic} works',
                    'Deep technical analysis of {topic}',
                    'Implementation details for {topic}'
                ],
                'best_for': ['implementation', 'debugging'],
                'parameters': {'breadth': 0.2, 'depth': 0.95}
            },
            {
                'id': 'comparative',
                'name': 'Comparative Analysis',
                'approach': 'Compare multiple approaches',
                'prompts': [
                    'Compare approaches to {topic}',
                    'Pros and cons of different {topic} methods',
                    'Trade-offs in {topic} implementations'
                ],
                'best_for': ['decision_making', 'optimization'],
                'parameters': {'breadth': 0.6, 'depth': 0.6}
            },
            {
                'id': 'pattern_mining',
                'name': 'Pattern Mining',
                'approach': 'Find patterns and best practices',
                'prompts': [
                    'Common patterns in {topic}',
                    'Best practices for {topic}',
                    'Recurring solutions to {topic}'
                ],
                'best_for': ['learning', 'improvement'],
                'parameters': {'breadth': 0.7, 'depth': 0.5}
            },
            {
                'id': 'innovative',
                'name': 'Innovation Search',
                'approach': 'Find cutting-edge approaches',
                'prompts': [
                    'Latest innovations in {topic}',
                    'Experimental approaches to {topic}',
                    'Future directions for {topic}'
                ],
                'best_for': ['research', 'advancement'],
                'parameters': {'breadth': 0.5, 'depth': 0.7, 'novelty': 0.9}
            }
        ]
    
    async def research_with_evolution(self,
                                     topic: str,
                                     goal: str,
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform research using evolved strategies.
        
        Args:
            topic: Research topic
            goal: Research goal
            context: Additional context
            
        Returns:
            Research results with meta-learning
        """
        print(f"Researching '{topic}' for goal: {goal}")
        
        # Select best strategy
        strategy = await self._select_optimal_strategy(topic, goal, context)
        
        # Adapt strategy based on history
        adapted_strategy = await self._adapt_strategy(strategy, topic, goal)
        
        # Execute research
        results = await self._execute_research(adapted_strategy, topic, goal, context)
        
        # Evaluate effectiveness
        effectiveness = await self._evaluate_research_effectiveness(
            results, goal, strategy['id']
        )
        
        # Update strategy scores
        self._update_strategy_effectiveness(strategy['id'], effectiveness)
        
        # Record for learning
        research_record = {
            'id': self._generate_research_id(),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'topic': topic,
            'goal': goal,
            'strategy_used': adapted_strategy,
            'results': results,
            'effectiveness': effectiveness
        }
        
        self.research_history.append(research_record)
        
        # Evolve strategies periodically
        if len(self.research_history) % 10 == 0:
            await self._evolve_strategies()
        
        return {
            'results': results,
            'strategy': adapted_strategy['name'],
            'effectiveness': effectiveness,
            'meta_insights': await self._generate_meta_insights(research_record)
        }
    
    async def _select_optimal_strategy(self,
                                      topic: str,
                                      goal: str,
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Select optimal research strategy using AI.
        
        Args:
            topic: Research topic
            goal: Research goal
            context: Additional context
            
        Returns:
            Selected strategy
        """
        # Get strategy recommendations from AI
        prompt = f"""
        Select the best research strategy for this task:
        
        Topic: {topic}
        Goal: {goal}
        Context: {json.dumps(context, indent=2) if context else 'None'}
        
        Available Strategies:
        {json.dumps([
            {
                'id': s['id'],
                'name': s['name'],
                'approach': s['approach'],
                'best_for': s['best_for'],
                'effectiveness': self.effectiveness_scores[s['id']]
            }
            for s in self.strategy_pool
        ], indent=2)}
        
        Consider:
        1. Match between strategy strengths and goal
        2. Historical effectiveness scores
        3. Topic characteristics
        4. Desired outcome type
        
        Return the strategy ID that best fits this research task.
        Format: {{"strategy_id": "...", "reasoning": "..."}}
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        selection = self._parse_json_response(response)
        
        # Find selected strategy
        strategy_id = selection.get('strategy_id', 'broad_survey')
        strategy = next(
            (s for s in self.strategy_pool if s['id'] == strategy_id),
            self.strategy_pool[0]
        )
        
        return strategy.copy()
    
    async def _adapt_strategy(self,
                             strategy: Dict[str, Any],
                             topic: str,
                             goal: str) -> Dict[str, Any]:
        """Adapt strategy based on learning.
        
        Args:
            strategy: Base strategy
            topic: Research topic
            goal: Research goal
            
        Returns:
            Adapted strategy
        """
        # Find similar past researches
        similar_researches = self._find_similar_researches(topic, goal)
        
        if not similar_researches:
            return strategy
        
        # Learn from past successes/failures
        prompt = f"""
        Adapt this research strategy based on past learnings:
        
        Base Strategy:
        {json.dumps(strategy, indent=2)}
        
        Current Research:
        - Topic: {topic}
        - Goal: {goal}
        
        Similar Past Researches:
        {json.dumps(similar_researches, indent=2)}
        
        Adapt the strategy by:
        1. Modifying prompts based on what worked
        2. Adjusting parameters (breadth, depth, etc.)
        3. Adding new prompt variations
        4. Incorporating successful patterns
        
        Return adapted strategy with improvements.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        adapted = self._parse_json_response(response)
        
        if adapted:
            # Merge adaptations with base strategy
            strategy.update(adapted)
            strategy['adapted'] = True
            strategy['adaptation_reason'] = adapted.get('reasoning', 'Learning from history')
        
        return strategy
    
    async def _execute_research(self,
                               strategy: Dict[str, Any],
                               topic: str,
                               goal: str,
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute research using strategy.
        
        Args:
            strategy: Research strategy
            topic: Research topic
            goal: Research goal
            context: Additional context
            
        Returns:
            Research results
        """
        results = {
            'findings': [],
            'insights': [],
            'recommendations': [],
            'sources': []
        }
        
        # Execute each prompt in strategy
        for prompt_template in strategy.get('prompts', []):
            prompt = prompt_template.format(topic=topic)
            
            # Add goal context
            full_prompt = f"""
            {prompt}
            
            Research Goal: {goal}
            {"Context: " + json.dumps(context, indent=2) if context else ""}
            
            Provide comprehensive, actionable findings.
            """
            
            response = await self.ai_brain.generate_enhanced_response(full_prompt)
            
            # Parse and categorize response
            parsed = await self._parse_research_response(response, goal)
            
            results['findings'].extend(parsed.get('findings', []))
            results['insights'].extend(parsed.get('insights', []))
            results['recommendations'].extend(parsed.get('recommendations', []))
        
        # Use AI brain for external research if available
        if self.ai_brain and hasattr(self.ai_brain, 'gather_context'):
            external_research = await self._gather_external_research(topic)
            results['sources'].extend(external_research)
        
        # Synthesize findings
        results['synthesis'] = await self._synthesize_findings(results, goal)
        
        return results
    
    async def _gather_external_research(self, topic: str) -> List[Dict[str, Any]]:
        """Gather external research on topic.
        
        Args:
            topic: Research topic
            
        Returns:
            External research findings
        """
        # This would use context gatherer's research capabilities
        # For now, return mock data
        return [{
            'source': 'external_research',
            'relevance': 'high',
            'summary': f'External findings on {topic}'
        }]
    
    async def _parse_research_response(self,
                                      response: Dict[str, Any],
                                      goal: str) -> Dict[str, Any]:
        """Parse and categorize research response.
        
        Args:
            response: AI response
            goal: Research goal
            
        Returns:
            Categorized findings
        """
        content = response.get('content', '')
        
        # Use AI to categorize
        prompt = f"""
        Categorize this research response based on the goal:
        
        Goal: {goal}
        Response: {content[:1000]}...
        
        Extract and categorize into:
        - findings: Factual discoveries
        - insights: Deeper understanding
        - recommendations: Actionable suggestions
        
        Format as JSON.
        """
        
        categorization = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(categorization)
    
    async def _synthesize_findings(self,
                                  results: Dict[str, Any],
                                  goal: str) -> Dict[str, Any]:
        """Synthesize research findings.
        
        Args:
            results: Research results
            goal: Research goal
            
        Returns:
            Synthesis
        """
        prompt = f"""
        Synthesize these research findings into actionable conclusions:
        
        Goal: {goal}
        
        Findings: {json.dumps(results['findings'][:5], indent=2)}
        Insights: {json.dumps(results['insights'][:3], indent=2)}
        Recommendations: {json.dumps(results['recommendations'][:3], indent=2)}
        
        Create synthesis with:
        - key_conclusions: Main takeaways
        - action_items: Specific next steps
        - confidence_level: How confident in findings (0.0-1.0)
        - gaps: What's still unknown
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    async def _evaluate_research_effectiveness(self,
                                             results: Dict[str, Any],
                                             goal: str,
                                             strategy_id: str) -> Dict[str, Any]:
        """Evaluate how effective the research was.
        
        Args:
            results: Research results
            goal: Research goal
            strategy_id: Strategy used
            
        Returns:
            Effectiveness evaluation
        """
        prompt = f"""
        Evaluate the effectiveness of this research:
        
        Goal: {goal}
        Strategy Used: {strategy_id}
        
        Results Summary:
        - Findings: {len(results.get('findings', []))}
        - Insights: {len(results.get('insights', []))}
        - Recommendations: {len(results.get('recommendations', []))}
        - Synthesis Quality: {json.dumps(results.get('synthesis', {}), indent=2)}
        
        Evaluate:
        1. Goal achievement (0.0-1.0)
        2. Finding quality (0.0-1.0)
        3. Actionability (0.0-1.0)
        4. Completeness (0.0-1.0)
        5. Overall effectiveness (0.0-1.0)
        
        Also identify what worked well and what could improve.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        evaluation = self._parse_json_response(response)
        
        # Default scores if parsing fails
        if not evaluation:
            evaluation = {
                'goal_achievement': 0.7,
                'finding_quality': 0.7,
                'actionability': 0.6,
                'completeness': 0.7,
                'overall_effectiveness': 0.7
            }
        
        return evaluation
    
    def _update_strategy_effectiveness(self, strategy_id: str, effectiveness: Dict[str, Any]):
        """Update strategy effectiveness scores.
        
        Args:
            strategy_id: Strategy identifier
            effectiveness: Effectiveness evaluation
        """
        current = self.effectiveness_scores[strategy_id]
        
        # Weighted average with recency bias
        new_score = effectiveness.get('overall_effectiveness', 0.5)
        current['uses'] += 1
        
        # More recent results have more weight
        weight = min(0.3, 1.0 / current['uses'])
        current['score'] = (1 - weight) * current['score'] + weight * new_score
        
        # Track specific strengths
        if 'strengths' not in current:
            current['strengths'] = []
        
        if effectiveness.get('goal_achievement', 0) > 0.8:
            current['strengths'].append('goal_achievement')
    
    async def _evolve_strategies(self):
        """Evolve research strategies based on performance."""
        print("Evolving research strategies...")
        
        # Analyze strategy performance
        performance_analysis = self._analyze_strategy_performance()
        
        # Generate new strategies
        new_strategies = await self._generate_evolved_strategies(performance_analysis)
        
        # Test new strategies
        for strategy in new_strategies:
            if await self._test_strategy_viability(strategy):
                self.strategy_pool.append(strategy)
                self.evolved_strategies.append(strategy)
        
        # Prune underperforming strategies
        self._prune_weak_strategies()
        
        # Generate meta-insights
        insights = await self._generate_evolution_insights()
        self.meta_insights.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'insights': insights,
            'strategy_count': len(self.strategy_pool)
        })
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance of all strategies.
        
        Returns:
            Performance analysis
        """
        analysis = {
            'by_strategy': {},
            'by_goal_type': defaultdict(list),
            'successful_patterns': [],
            'failure_patterns': []
        }
        
        # Analyze each strategy
        for strategy_id, scores in self.effectiveness_scores.items():
            analysis['by_strategy'][strategy_id] = {
                'score': scores['score'],
                'uses': scores['uses'],
                'strengths': scores.get('strengths', [])
            }
        
        # Analyze by goal patterns
        for record in self.research_history[-50:]:  # Last 50 researches
            goal_type = self._categorize_goal(record['goal'])
            effectiveness = record['effectiveness'].get('overall_effectiveness', 0)
            
            analysis['by_goal_type'][goal_type].append({
                'strategy': record['strategy_used']['id'],
                'effectiveness': effectiveness
            })
        
        return analysis
    
    async def _generate_evolved_strategies(self,
                                          performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate new evolved strategies.
        
        Args:
            performance: Performance analysis
            
        Returns:
            New strategies
        """
        prompt = f"""
        Generate evolved research strategies based on performance data:
        
        Performance Analysis:
        {json.dumps(performance, indent=2)}
        
        Current Best Strategies:
        {json.dumps([
            s for s in self.strategy_pool 
            if self.effectiveness_scores[s['id']]['score'] > 0.7
        ], indent=2)}
        
        Generate 2-3 new strategies that:
        1. Combine strengths of successful strategies
        2. Address weaknesses in current approaches
        3. Introduce novel research patterns
        4. Target specific goal types that need improvement
        
        Format each strategy like existing ones with:
        - id, name, approach, prompts, best_for, parameters
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        new_strategies = self._parse_json_response(response)
        
        if isinstance(new_strategies, list):
            return new_strategies
        
        return []
    
    async def _test_strategy_viability(self, strategy: Dict[str, Any]) -> bool:
        """Test if strategy is viable.
        
        Args:
            strategy: Strategy to test
            
        Returns:
            Viability status
        """
        # Simple viability check
        required_fields = ['id', 'name', 'approach', 'prompts']
        
        for field in required_fields:
            if field not in strategy:
                return False
        
        # Ensure unique ID
        if any(s['id'] == strategy['id'] for s in self.strategy_pool):
            strategy['id'] = f"{strategy['id']}_{len(self.evolved_strategies)}"
        
        return True
    
    def _prune_weak_strategies(self):
        """Remove underperforming strategies."""
        # Keep at least 5 strategies
        if len(self.strategy_pool) <= 5:
            return
        
        # Remove strategies with poor performance and sufficient tests
        to_remove = []
        
        for strategy in self.strategy_pool:
            score_data = self.effectiveness_scores[strategy['id']]
            
            if score_data['uses'] >= 5 and score_data['score'] < 0.4:
                to_remove.append(strategy)
        
        for strategy in to_remove:
            self.strategy_pool.remove(strategy)
            print(f"Pruned strategy: {strategy['name']} (score: {score_data['score']:.2f})")
    
    async def _generate_evolution_insights(self) -> Dict[str, Any]:
        """Generate insights from strategy evolution.
        
        Returns:
            Evolution insights
        """
        prompt = f"""
        Generate insights from research strategy evolution:
        
        Strategy Pool Size: {len(self.strategy_pool)}
        Evolved Strategies: {len(self.evolved_strategies)}
        
        Performance Trends:
        {json.dumps({
            sid: {'score': s['score'], 'uses': s['uses']}
            for sid, s in self.effectiveness_scores.items()
        }, indent=2)}
        
        Provide insights on:
        1. Which research approaches work best
        2. Emerging patterns in effective research
        3. Gaps in current strategies
        4. Recommendations for future evolution
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    def _find_similar_researches(self, topic: str, goal: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar past researches.
        
        Args:
            topic: Current topic
            goal: Current goal
            limit: Maximum results
            
        Returns:
            Similar researches
        """
        similar = []
        
        for record in reversed(self.research_history):
            # Simple similarity based on topic/goal overlap
            topic_similarity = self._calculate_text_similarity(topic, record['topic'])
            goal_similarity = self._calculate_text_similarity(goal, record['goal'])
            
            combined_similarity = (topic_similarity + goal_similarity) / 2
            
            if combined_similarity > 0.3:
                similar.append({
                    'topic': record['topic'],
                    'goal': record['goal'],
                    'strategy': record['strategy_used']['name'],
                    'effectiveness': record['effectiveness'].get('overall_effectiveness', 0),
                    'key_findings': record['results'].get('synthesis', {}).get('key_conclusions', [])[:2]
                })
            
            if len(similar) >= limit:
                break
        
        return similar
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score 0.0-1.0
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _categorize_goal(self, goal: str) -> str:
        """Categorize research goal.
        
        Args:
            goal: Goal text
            
        Returns:
            Goal category
        """
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ['implement', 'build', 'create']):
            return 'implementation'
        elif any(word in goal_lower for word in ['understand', 'learn', 'explore']):
            return 'learning'
        elif any(word in goal_lower for word in ['fix', 'solve', 'debug']):
            return 'problem_solving'
        elif any(word in goal_lower for word in ['improve', 'optimize', 'enhance']):
            return 'optimization'
        else:
            return 'general'
    
    def _generate_research_id(self) -> str:
        """Generate unique research ID.
        
        Returns:
            Research ID
        """
        content = f"{datetime.now().isoformat()}{len(self.research_history)}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def _generate_meta_insights(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Generate meta-insights from research.
        
        Args:
            record: Research record
            
        Returns:
            Meta insights
        """
        # Quick insights based on effectiveness
        effectiveness = record['effectiveness'].get('overall_effectiveness', 0)
        
        insights = {
            'strategy_fit': 'good' if effectiveness > 0.7 else 'poor',
            'learning_value': 'high' if len(record['results'].get('insights', [])) > 2 else 'low'
        }
        
        return insights
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Any:
        """Parse JSON from AI response."""
        content = response.get('content', '')
        
        try:
            import re
            
            # Look for JSON array
            array_match = re.search(r'\[[\s\S]*\]', content)
            if array_match:
                return json.loads(array_match.group())
            
            # Look for JSON object
            obj_match = re.search(r'\{[\s\S]*\}', content)
            if obj_match:
                return json.loads(obj_match.group())
                
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error parsing JSON: {e}")
        
        return {}
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of research evolution.
        
        Returns:
            Evolution summary
        """
        return {
            'total_researches': len(self.research_history),
            'strategy_pool_size': len(self.strategy_pool),
            'evolved_strategies': len(self.evolved_strategies),
            'effectiveness_scores': dict(self.effectiveness_scores),
            'latest_insights': self.meta_insights[-1] if self.meta_insights else None,
            'top_strategies': self._get_top_strategies()
        }
    
    def _get_top_strategies(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get top performing strategies.
        
        Args:
            n: Number of top strategies
            
        Returns:
            Top strategies
        """
        ranked = sorted(
            self.strategy_pool,
            key=lambda s: self.effectiveness_scores[s['id']]['score'],
            reverse=True
        )
        
        return [
            {
                'name': s['name'],
                'score': self.effectiveness_scores[s['id']]['score'],
                'uses': self.effectiveness_scores[s['id']]['uses']
            }
            for s in ranked[:n]
        ]


async def demonstrate_research_evolver():
    """Demonstrate research evolution."""
    print("=== Research Evolver Demo ===\n")
    
    # Mock AI brain
    class MockAIBrain:
        async def generate_enhanced_response(self, prompt):
            # Return different responses based on prompt content
            if "Select the best research strategy" in prompt:
                return {'content': '{"strategy_id": "deep_dive", "reasoning": "Best for implementation"}'}
            elif "Categorize this research response" in prompt:
                return {'content': '{"findings": ["Found A", "Found B"], "insights": ["Insight 1"]}'}
            else:
                return {'content': '{"overall_effectiveness": 0.85}'}
    
    ai_brain = MockAIBrain()
    evolver = ResearchEvolver(ai_brain, None)
    
    # Perform evolved research
    print("Performing research with evolution...")
    result = await evolver.research_with_evolution(
        topic="AI self-improvement architectures",
        goal="implement modular self-improvement system",
        context={"domain": "software_engineering"}
    )
    
    print(f"\nResearch completed:")
    print(f"- Strategy used: {result['strategy']}")
    print(f"- Effectiveness: {result['effectiveness']}")
    print(f"- Meta insights: {result['meta_insights']}")
    
    # Show evolution summary
    print("\n=== Evolution Summary ===")
    summary = evolver.get_evolution_summary()
    print(f"Total researches: {summary['total_researches']}")
    print(f"Strategy pool size: {summary['strategy_pool_size']}")
    print("\nTop strategies:")
    for strategy in summary['top_strategies']:
        print(f"- {strategy['name']}: {strategy['score']:.2f} (used {strategy['uses']} times)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_research_evolver())