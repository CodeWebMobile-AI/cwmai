"""
Outcome-Based Learning System

Learns what creates value from actual outcomes without any hardcoded values.
All value assessments are done through AI reasoning.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class OutcomeLearningSystem:
    """Learn from outcomes without hardcoded values - pure AI reasoning."""
    
    def __init__(self, ai_brain):
        """Initialize learning system with AI brain.
        
        Args:
            ai_brain: AI brain for intelligent analysis
        """
        self.ai_brain = ai_brain
        self.outcome_history = []
        self.value_patterns = {}
        self.learning_insights = []
        self.logger = logging.getLogger(__name__)
        
    async def record_outcome(self, task: Dict[str, Any], 
                           outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Record task outcome and learn from it.
        
        Args:
            task: The executed task
            outcome: The execution outcome
            
        Returns:
            Learning record with value assessment
        """
        self.logger.info(f"Recording outcome for task: {task.get('title', 'Unknown')}")
        
        # Assess value using AI
        value_assessment = await self.assess_outcome_value(task, outcome)
        
        # Create comprehensive record
        record = {
            'id': f"outcome_{len(self.outcome_history)}_{datetime.now(timezone.utc).timestamp()}",
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task': task,
            'outcome': outcome,
            'value_assessment': value_assessment,
            'context': await self._capture_context()
        }
        
        # Add to history
        self.outcome_history.append(record)
        
        # Update patterns
        await self._update_value_patterns(record)
        
        # Generate learning insights
        if len(self.outcome_history) % 5 == 0:  # Every 5 outcomes
            await self._generate_learning_insights()
            
        return record
    
    async def assess_outcome_value(self, task: Dict[str, Any], 
                                  outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Assess value created by task outcome using AI reasoning.
        
        Args:
            task: The task that was executed
            outcome: The execution outcome
            
        Returns:
            Value assessment with score and reasoning
        """
        # Get similar past outcomes for context
        similar_outcomes = self._find_similar_outcomes(task)
        
        prompt = f"""
        Assess the value created by this task outcome for an AI development orchestrator.
        
        Task Details:
        {json.dumps(task, indent=2)}
        
        Execution Outcome:
        {json.dumps(outcome, indent=2)}
        
        Similar Past Outcomes for Context:
        {json.dumps(similar_outcomes, indent=2)}
        
        Consider these value dimensions:
        1. MISSION ADVANCEMENT: How much did this advance the system's mission of building a software portfolio?
        2. TANGIBLE OUTPUT: What concrete deliverables were created? (repos, features, improvements)
        3. LEARNING VALUE: What did the system learn that will improve future performance?
        4. EFFICIENCY: Was this an efficient use of resources?
        5. STRATEGIC VALUE: Does this position the system better for future success?
        6. QUALITY: Was the output high quality and sustainable?
        
        For NEW_PROJECT tasks: Consider if a real, working application was created from the Laravel React starter kit.
        For FEATURE tasks: Consider if it enhanced an existing project meaningfully.
        For improvement tasks: Consider if the AI system became more capable.
        
        Provide assessment as JSON with:
        - value_score: 0.0 to 1.0 (where 1.0 is maximum value)
        - dimension_scores: Score for each dimension above (0.0 to 1.0)
        - reasoning: Detailed explanation of the score
        - key_learnings: What we learned from this outcome
        - improvement_suggestions: How similar tasks could create more value
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        assessment = self._parse_value_assessment(response)
        
        # Ensure we have a valid assessment
        if 'value_score' not in assessment:
            assessment['value_score'] = 0.5  # Default middle score
            assessment['reasoning'] = 'Unable to assess - using default score'
            
        return assessment
    
    def _find_similar_outcomes(self, task: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar past outcomes for context.
        
        Args:
            task: Current task
            limit: Maximum number of similar outcomes
            
        Returns:
            List of similar outcome summaries
        """
        similar = []
        task_type = task.get('type')
        task_domain = self._extract_domain(task)
        
        # Look through recent history
        for record in reversed(self.outcome_history[-20:]):
            past_task = record.get('task', {})
            
            # Check similarity
            if past_task.get('type') == task_type:
                similarity_score = self._calculate_similarity(task, past_task)
                
                similar.append({
                    'task_title': past_task.get('title'),
                    'task_type': past_task.get('type'),
                    'success': record.get('outcome', {}).get('success', False),
                    'value_score': record.get('value_assessment', {}).get('value_score', 0),
                    'reasoning_summary': record.get('value_assessment', {}).get('reasoning', '')[:200],
                    'similarity_score': similarity_score
                })
                
        # Sort by similarity and return top matches
        similar.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similar[:limit]
    
    def _calculate_similarity(self, task1: Dict[str, Any], task2: Dict[str, Any]) -> float:
        """Calculate similarity between two tasks.
        
        Args:
            task1: First task
            task2: Second task
            
        Returns:
            Similarity score 0.0 to 1.0
        """
        score = 0.0
        
        # Type match
        if task1.get('type') == task2.get('type'):
            score += 0.4
            
        # Title similarity (simple word overlap)
        title1_words = set(task1.get('title', '').lower().split())
        title2_words = set(task2.get('title', '').lower().split())
        if title1_words and title2_words:
            overlap = len(title1_words.intersection(title2_words))
            total = len(title1_words.union(title2_words))
            score += 0.3 * (overlap / total if total > 0 else 0)
            
        # Domain similarity
        if self._extract_domain(task1) == self._extract_domain(task2):
            score += 0.3
            
        return min(score, 1.0)
    
    def _extract_domain(self, task: Dict[str, Any]) -> str:
        """Extract domain from task (auth, api, ui, etc).
        
        Args:
            task: Task to analyze
            
        Returns:
            Domain string
        """
        text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        
        domains = {
            'auth': ['auth', 'login', '2fa', 'oauth', 'security'],
            'api': ['api', 'rest', 'graphql', 'endpoint'],
            'ui': ['ui', 'frontend', 'dashboard', 'interface'],
            'data': ['database', 'data', 'analytics', 'report'],
            'infra': ['deploy', 'ci/cd', 'docker', 'kubernetes']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in text for keyword in keywords):
                return domain
                
        return 'general'
    
    async def _update_value_patterns(self, record: Dict[str, Any]) -> None:
        """Update value patterns based on new outcome.
        
        Args:
            record: Outcome record
        """
        task_type = record['task'].get('type', 'unknown')
        value_score = record['value_assessment'].get('value_score', 0)
        
        # Initialize pattern tracking if needed
        if task_type not in self.value_patterns:
            self.value_patterns[task_type] = {
                'outcomes': [],
                'average_value': 0,
                'success_rate': 0,
                'patterns': []
            }
            
        # Add outcome
        self.value_patterns[task_type]['outcomes'].append({
            'value_score': value_score,
            'success': record['outcome'].get('success', False),
            'timestamp': record['timestamp']
        })
        
        # Recalculate metrics
        outcomes = self.value_patterns[task_type]['outcomes']
        self.value_patterns[task_type]['average_value'] = np.mean([o['value_score'] for o in outcomes])
        self.value_patterns[task_type]['success_rate'] = np.mean([1 if o['success'] else 0 for o in outcomes])
        
        # Identify patterns every 10 outcomes
        if len(outcomes) % 10 == 0:
            await self._identify_patterns(task_type)
    
    async def _identify_patterns(self, task_type: str) -> None:
        """Identify patterns in outcomes for a task type.
        
        Args:
            task_type: Type of task to analyze
        """
        outcomes = self.value_patterns[task_type]['outcomes']
        recent_records = [r for r in self.outcome_history if r['task'].get('type') == task_type][-10:]
        
        prompt = f"""
        Identify patterns in these {task_type} task outcomes:
        
        Recent Outcomes:
        {json.dumps(recent_records, indent=2)}
        
        Metrics:
        - Average Value Score: {self.value_patterns[task_type]['average_value']:.2f}
        - Success Rate: {self.value_patterns[task_type]['success_rate']:.2%}
        
        Identify:
        1. What characteristics lead to high-value outcomes?
        2. What patterns cause failures or low value?
        3. What specific improvements would increase value?
        4. Are there timing patterns (when tasks succeed/fail)?
        
        Return as JSON with:
        - success_patterns: List of patterns that lead to success
        - failure_patterns: List of patterns that lead to failure
        - recommendations: Specific actionable recommendations
        - insights: Key insights about this task type
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        patterns = self._parse_json_response(response)
        
        self.value_patterns[task_type]['patterns'] = patterns
    
    async def _generate_learning_insights(self) -> None:
        """Generate high-level learning insights from all outcomes."""
        prompt = f"""
        Analyze the complete outcome history to generate learning insights:
        
        Outcome Summary:
        {json.dumps(self._get_outcome_summary(), indent=2)}
        
        Value Patterns by Type:
        {json.dumps(self.value_patterns, indent=2)}
        
        Generate insights about:
        1. Which types of tasks create the most value and why
        2. What patterns consistently lead to success
        3. What the system should do more of
        4. What the system should avoid
        5. How task generation could be improved
        6. Strategic recommendations for the system
        
        Format as JSON with:
        - key_insights: List of important discoveries
        - do_more: Specific things to do more of
        - do_less: Specific things to avoid
        - strategic_recommendations: High-level strategy adjustments
        - task_generation_improvements: How to generate better tasks
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        insights = self._parse_json_response(response)
        
        self.learning_insights.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'insights': insights,
            'outcome_count': len(self.outcome_history)
        })
    
    def _get_outcome_summary(self) -> Dict[str, Any]:
        """Get summary of all outcomes for analysis.
        
        Returns:
            Summary statistics
        """
        if not self.outcome_history:
            return {'total': 0, 'by_type': {}, 'overall_success_rate': 0}
            
        total = len(self.outcome_history)
        successful = sum(1 for r in self.outcome_history if r['outcome'].get('success', False))
        
        by_type = {}
        for record in self.outcome_history:
            task_type = record['task'].get('type', 'unknown')
            if task_type not in by_type:
                by_type[task_type] = {'count': 0, 'successes': 0, 'total_value': 0}
                
            by_type[task_type]['count'] += 1
            if record['outcome'].get('success', False):
                by_type[task_type]['successes'] += 1
            by_type[task_type]['total_value'] += record['value_assessment'].get('value_score', 0)
            
        return {
            'total': total,
            'successful': successful,
            'overall_success_rate': successful / total if total > 0 else 0,
            'by_type': by_type
        }
    
    async def get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations based on all learning.
        
        Returns:
            Recommendations for system improvement
        """
        if len(self.outcome_history) < 3:
            return {
                'status': 'insufficient_data',
                'message': 'Need more outcomes to generate recommendations'
            }
            
        # Get latest insights
        latest_insights = self.learning_insights[-1]['insights'] if self.learning_insights else {}
        
        prompt = f"""
        Based on all learning, provide actionable recommendations:
        
        Learning Summary:
        - Total Outcomes: {len(self.outcome_history)}
        - Value Patterns: {json.dumps(self.value_patterns, indent=2)}
        - Latest Insights: {json.dumps(latest_insights, indent=2)}
        
        Provide specific, actionable recommendations for:
        1. What types of tasks to prioritize
        2. How to improve task generation
        3. Which patterns to replicate
        4. What to avoid
        5. Strategic pivots if needed
        
        Format as JSON with:
        - immediate_actions: Things to do right now
        - task_priorities: Which task types to focus on
        - avoid_list: Specific things to stop doing
        - success_template: Template for high-value tasks
        - strategic_shift: Any major strategy changes needed
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    async def predict_task_value(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the potential value of a task before execution.
        
        Args:
            task: Proposed task
            
        Returns:
            Predicted value and confidence
        """
        similar_outcomes = self._find_similar_outcomes(task, limit=10)
        
        prompt = f"""
        Predict the potential value of this proposed task:
        
        Proposed Task:
        {json.dumps(task, indent=2)}
        
        Similar Past Outcomes:
        {json.dumps(similar_outcomes, indent=2)}
        
        Current Success Patterns:
        {json.dumps(self.value_patterns.get(task.get('type', 'unknown'), {}).get('patterns', {}), indent=2)}
        
        Predict:
        1. Likely value score (0.0 to 1.0)
        2. Success probability
        3. Potential risks
        4. Ways to increase value
        5. Whether this task should be executed
        
        Format as JSON with:
        - predicted_value: 0.0 to 1.0
        - success_probability: 0.0 to 1.0
        - confidence: How confident in this prediction (0.0 to 1.0)
        - risks: List of potential risks
        - value_improvements: Ways to modify task for more value
        - recommendation: 'execute', 'modify', or 'skip'
        - reasoning: Explanation of prediction
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    async def _capture_context(self) -> Dict[str, Any]:
        """Capture current context for outcome record.
        
        Returns:
            Context snapshot
        """
        return {
            'total_outcomes': len(self.outcome_history),
            'recent_success_rate': self._calculate_recent_success_rate(),
            'active_patterns': len(self.value_patterns)
        }
    
    def _calculate_recent_success_rate(self, window: int = 10) -> float:
        """Calculate success rate for recent outcomes.
        
        Args:
            window: Number of recent outcomes to consider
            
        Returns:
            Success rate
        """
        recent = self.outcome_history[-window:] if len(self.outcome_history) >= window else self.outcome_history
        if not recent:
            return 0.0
            
        successful = sum(1 for r in recent if r['outcome'].get('success', False))
        return successful / len(recent)
    
    def _parse_value_assessment(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse value assessment from AI response.
        
        Args:
            response: AI response
            
        Returns:
            Parsed assessment
        """
        return self._parse_json_response(response)
    
    def _parse_json_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON from AI response.
        
        Args:
            response: AI response
            
        Returns:
            Parsed JSON or empty dict
        """
        content = response.get('content', '')
        
        try:
            import re
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            
        return {}
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learning.
        
        Returns:
            Learning summary
        """
        return {
            'total_outcomes': len(self.outcome_history),
            'value_patterns': self.value_patterns,
            'latest_insights': self.learning_insights[-1] if self.learning_insights else None,
            'recent_success_rate': self._calculate_recent_success_rate(),
            'high_value_task_types': self._get_high_value_types()
        }
    
    def _get_high_value_types(self) -> List[str]:
        """Get task types with highest average value.
        
        Returns:
            List of high-value task types
        """
        type_values = [
            (task_type, patterns['average_value'])
            for task_type, patterns in self.value_patterns.items()
            if patterns['outcomes']
        ]
        
        type_values.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in type_values[:3]]  # Top 3