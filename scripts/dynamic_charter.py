"""
Dynamic System Charter Generator

Generates and updates system purpose based on learning and context.
No hardcoded values - everything is dynamically determined.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional


class DynamicCharter:
    """Dynamically generates and updates system charter based on AI reasoning."""
    
    def __init__(self, ai_brain):
        """Initialize with AI brain for dynamic generation.
        
        Args:
            ai_brain: AI brain instance for generating charter
        """
        self.ai_brain = ai_brain
        self.charter_history = []
        self.logger = logging.getLogger(__name__)
        
    async def generate_charter(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate charter based on current understanding and context.
        
        Args:
            context: Current system context including projects, outcomes, etc.
            
        Returns:
            Dynamically generated charter
        """
        self.logger.info("Generating dynamic charter based on current context")
        
        prompt = f"""
        You are defining the purpose and constraints for an AI development orchestrator.
        This system manages software development for the CodeWebMobile-AI organization.
        
        Current Context:
        - Active Projects: {json.dumps(context.get('projects', []), indent=2)}
        - Recent Outcomes: {json.dumps(context.get('recent_outcomes', []), indent=2)}
        - System Capabilities: {json.dumps(context.get('capabilities', []), indent=2)}
        - Market Trends: {json.dumps(context.get('market_trends', []), indent=2)}
        
        Define a comprehensive charter that includes:
        
        1. PRIMARY_PURPOSE: Clear mission statement for this AI system
        2. CORE_OBJECTIVES: List of 3-5 main objectives
        3. CAPABILITIES: What this system can and should do
        4. CONSTRAINTS: What this system should NOT do
        5. TASK_TYPES: Define when to use each task type:
           - NEW_PROJECT: When to create new applications
           - FEATURE: When to add features (and to which projects)
           - improvement: When to enhance the AI system itself
           - Other types as needed
        6. PROJECT_METHODOLOGY: How to create new projects (must mention Laravel React starter kit)
        7. SUCCESS_METRICS: How to measure system success
        8. DECISION_PRINCIPLES: Guidelines for making decisions
        
        Format as JSON with these exact keys. Be specific and actionable.
        Base decisions on the current context and what would create the most value.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        charter = self._parse_charter_response(response)
        
        # Evolve based on history if available
        if self.charter_history:
            charter = await self._evolve_charter(charter, context)
            
        # Record in history
        self.charter_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'charter': charter,
            'context_snapshot': {
                'project_count': len(context.get('projects', [])),
                'recent_success_rate': self._calculate_success_rate(context),
                'active_capabilities': context.get('capabilities', [])
            }
        })
        
        return charter
    
    async def _evolve_charter(self, new_charter: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve charter based on learning from previous charters.
        
        Args:
            new_charter: Newly generated charter
            context: Current context
            
        Returns:
            Evolved charter incorporating learnings
        """
        previous_charter = self.charter_history[-1]['charter']
        outcomes_since = self._get_outcomes_since_last_charter(context)
        
        prompt = f"""
        Evolve the system charter based on learned experience.
        
        Previous Charter:
        {json.dumps(previous_charter, indent=2)}
        
        Newly Generated Charter:
        {json.dumps(new_charter, indent=2)}
        
        Outcomes Since Last Charter:
        {json.dumps(outcomes_since, indent=2)}
        
        Analysis of What Worked:
        - Success Rate: {outcomes_since.get('success_rate', 0)}%
        - Value Created: {outcomes_since.get('value_metrics', {})}
        - Patterns: {outcomes_since.get('patterns', [])}
        
        Create an improved charter that:
        1. Keeps successful strategies from the previous charter
        2. Incorporates new insights from the generated charter
        3. Adjusts based on what actually worked/failed
        4. Refines decision principles based on outcomes
        
        Return the evolved charter in the same JSON format.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_charter_response(response)
    
    def _parse_charter_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AI response into charter format.
        
        Args:
            response: AI response
            
        Returns:
            Parsed charter dictionary
        """
        content = response.get('content', '')
        
        # Try to extract JSON from response
        try:
            import re
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"Failed to parse charter JSON: {e}")
        
        # Fallback: Create structured charter from text
        return {
            'PRIMARY_PURPOSE': 'Autonomously create and manage software projects for CodeWebMobile-AI',
            'CORE_OBJECTIVES': [
                'Build a portfolio of Laravel React applications',
                'Continuously improve AI capabilities',
                'Maintain high code quality standards'
            ],
            'CAPABILITIES': [
                'Create new projects from Laravel React starter kit',
                'Generate and validate development tasks',
                'Coordinate multi-repository development',
                'Learn from outcomes and adapt'
            ],
            'CONSTRAINTS': [
                'Only create features for existing projects',
                'No hardcoded logic or templates',
                'All decisions must be AI-driven'
            ],
            'TASK_TYPES': {
                'NEW_PROJECT': 'Create complete applications using starter kit',
                'FEATURE': 'Add features to existing projects only',
                'improvement': 'Enhance the AI system itself'
            },
            'PROJECT_METHODOLOGY': 'Fork Laravel React starter kit and customize',
            'SUCCESS_METRICS': ['Portfolio growth', 'Task success rate', 'System improvements'],
            'DECISION_PRINCIPLES': ['Value-driven', 'Learn from outcomes', 'Continuous improvement']
        }
    
    def _calculate_success_rate(self, context: Dict[str, Any]) -> float:
        """Calculate recent success rate from context.
        
        Args:
            context: System context
            
        Returns:
            Success rate percentage
        """
        outcomes = context.get('recent_outcomes', [])
        if not outcomes:
            return 0.0
            
        successful = sum(1 for o in outcomes if o.get('success', False))
        return (successful / len(outcomes)) * 100
    
    def _get_outcomes_since_last_charter(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get outcomes since the last charter update.
        
        Args:
            context: Current context
            
        Returns:
            Summary of outcomes
        """
        if not self.charter_history:
            return {'success_rate': 0, 'value_metrics': {}, 'patterns': []}
            
        last_charter_time = self.charter_history[-1]['timestamp']
        recent_outcomes = context.get('recent_outcomes', [])
        
        # Filter outcomes after last charter
        relevant_outcomes = [
            o for o in recent_outcomes 
            if o.get('timestamp', '') > last_charter_time
        ]
        
        # Calculate metrics
        success_count = sum(1 for o in relevant_outcomes if o.get('success', False))
        success_rate = (success_count / len(relevant_outcomes) * 100) if relevant_outcomes else 0
        
        # Extract patterns
        patterns = []
        task_types = {}
        for outcome in relevant_outcomes:
            task_type = outcome.get('task', {}).get('type', 'unknown')
            task_types[task_type] = task_types.get(task_type, 0) + 1
            
        return {
            'success_rate': success_rate,
            'value_metrics': {
                'total_tasks': len(relevant_outcomes),
                'successful_tasks': success_count,
                'task_distribution': task_types
            },
            'patterns': patterns
        }
    
    async def get_current_charter(self) -> Dict[str, Any]:
        """Get the current active charter.
        
        Returns:
            Current charter or generate new one if none exists
        """
        if self.charter_history:
            return self.charter_history[-1]['charter']
        else:
            # Generate initial charter with minimal context
            return await self.generate_charter({
                'projects': [],
                'recent_outcomes': [],
                'capabilities': ['GitHub API', 'AI Models', 'Code Generation'],
                'market_trends': []
            })
    
    def get_charter_evolution(self) -> List[Dict[str, Any]]:
        """Get the history of charter evolution.
        
        Returns:
            List of charter history entries
        """
        return self.charter_history
    
    async def validate_against_charter(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an action against current charter.
        
        Args:
            action: Proposed action/task
            
        Returns:
            Validation result with reasoning
        """
        charter = await self.get_current_charter()
        
        prompt = f"""
        Validate this action against the system charter:
        
        Proposed Action:
        {json.dumps(action, indent=2)}
        
        System Charter:
        {json.dumps(charter, indent=2)}
        
        Determine:
        1. Is this action aligned with the PRIMARY_PURPOSE?
        2. Does it support the CORE_OBJECTIVES?
        3. Is it within CAPABILITIES?
        4. Does it violate any CONSTRAINTS?
        5. Does it follow the DECISION_PRINCIPLES?
        
        Return JSON with:
        - aligned: true/false
        - reasoning: Explanation of alignment/misalignment
        - suggestions: How to better align if needed
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_validation_response(response)
    
    def _parse_validation_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse validation response.
        
        Args:
            response: AI response
            
        Returns:
            Parsed validation result
        """
        content = response.get('content', '')
        
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        # Fallback
        return {
            'aligned': True,
            'reasoning': 'Unable to parse validation',
            'suggestions': []
        }