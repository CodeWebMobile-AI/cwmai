"""
Intelligent Task Generator

Generates tasks based on dynamic goals and learning.
No templates, no hardcoded values - pure AI reasoning.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import asyncio


class IntelligentTaskGenerator:
    """Generate tasks with zero hardcoded logic - all AI-driven."""
    
    def __init__(self, ai_brain, charter_system, learning_system=None):
        """Initialize with AI brain and charter system.
        
        Args:
            ai_brain: AI brain for task generation
            charter_system: Dynamic charter system for guidelines
            learning_system: Optional learning system for value prediction
        """
        self.ai_brain = ai_brain
        self.charter_system = charter_system
        self.learning_system = learning_system
        self.task_history = []
        self.generation_patterns = {}
        self.logger = logging.getLogger(__name__)
        
    async def generate_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a task based on current needs and context.
        
        Args:
            context: Current system context
            
        Returns:
            Generated task
        """
        self.logger.info("Generating intelligent task based on context")
        
        # Get current charter for guidelines
        charter = await self.charter_system.get_current_charter()
        
        # Analyze what the system needs
        need_analysis = await self._analyze_system_needs(context, charter)
        
        # Generate task addressing the need
        task = await self._create_task_for_need(need_analysis, context, charter)
        
        # Ensure Laravel React starter kit for new projects
        if task.get('type') == 'NEW_PROJECT':
            task = await self._ensure_starter_kit_integration(task)
            
        # Apply learning if available
        if self.learning_system:
            task = await self._apply_learned_improvements(task)
            
        # Predict value if possible
        if self.learning_system:
            prediction = await self.learning_system.predict_task_value(task)
            if prediction.get('recommendation') == 'modify':
                task = await self._modify_for_higher_value(task, prediction)
                
        # Record generation
        self._record_generation(task, need_analysis)
        
        return task
    
    async def _analyze_system_needs(self, context: Dict[str, Any], 
                                   charter: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze what the system needs right now.
        
        Args:
            context: System context
            charter: Current charter
            
        Returns:
            Need analysis
        """
        prompt = f"""
        Analyze the current state of the AI development orchestrator system.
        
        System Charter:
        {json.dumps(charter, indent=2)}
        
        Current Context:
        - Active Projects: {json.dumps(context.get('projects', []), indent=2)}
        - Recent Tasks: {json.dumps(context.get('recent_tasks', [])[-5:], indent=2)}
        - System Capabilities: {json.dumps(context.get('capabilities', []), indent=2)}
        - Market Trends: {json.dumps(context.get('market_trends', [])[:3], indent=2)}
        - Recent Outcomes: {json.dumps(context.get('recent_outcomes', [])[-3:], indent=2)}
        
        Portfolio Analysis:
        - Total Projects: {len(context.get('projects', []))}
        - Project Types: {self._analyze_project_types(context.get('projects', []))}
        - Coverage Gaps: {self._identify_portfolio_gaps(context.get('projects', []))}
        
        Determine what the system should focus on next:
        1. Does it need to expand the portfolio with a new project?
        2. Should it enhance existing projects with features?
        3. Does the AI system itself need improvements?
        4. Are there maintenance or quality issues to address?
        
        Consider:
        - Portfolio balance and gaps
        - Market opportunities from trends
        - System performance and capabilities
        - Strategic objectives from charter
        
        Return analysis as JSON with:
        - need_type: 'portfolio_expansion', 'project_enhancement', 'system_improvement', 'quality_focus'
        - specific_need: Detailed description of what's needed
        - priority: 'critical', 'high', 'medium', 'low'
        - rationale: Why this is needed now
        - opportunity: What opportunity this addresses
        - suggested_approach: How to address this need
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    def _analyze_project_types(self, projects: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze types of projects in portfolio.
        
        Args:
            projects: List of projects
            
        Returns:
            Project type counts
        """
        types = {}
        for project in projects:
            # Extract type from project name/description
            project_type = self._categorize_project(project)
            types[project_type] = types.get(project_type, 0) + 1
        return types
    
    def _categorize_project(self, project: Dict[str, Any]) -> str:
        """Categorize a project based on its characteristics.
        
        Args:
            project: Project details
            
        Returns:
            Project category
        """
        name = project.get('name', '').lower()
        description = project.get('description', '').lower()
        text = f"{name} {description}"
        
        # Simple categorization
        if any(word in text for word in ['auth', 'login', '2fa', 'oauth']):
            return 'authentication'
        elif any(word in text for word in ['api', 'rest', 'graphql']):
            return 'api_service'
        elif any(word in text for word in ['dashboard', 'admin', 'panel']):
            return 'dashboard'
        elif any(word in text for word in ['analytics', 'report', 'data']):
            return 'analytics'
        else:
            return 'general'
    
    def _identify_portfolio_gaps(self, projects: List[Dict[str, Any]]) -> List[str]:
        """Identify gaps in the project portfolio.
        
        Args:
            projects: Current projects
            
        Returns:
            List of identified gaps
        """
        existing_types = set(self._categorize_project(p) for p in projects)
        
        # Common project types for a complete portfolio
        desired_types = {
            'authentication', 'api_service', 'dashboard', 'analytics',
            'notification', 'payment', 'cms', 'mobile_backend'
        }
        
        gaps = list(desired_types - existing_types)
        return gaps[:3]  # Top 3 gaps
    
    async def _create_task_for_need(self, need: Dict[str, Any], 
                                   context: Dict[str, Any],
                                   charter: Dict[str, Any]) -> Dict[str, Any]:
        """Create specific task for identified need.
        
        Args:
            need: Need analysis
            context: System context
            charter: System charter
            
        Returns:
            Generated task
        """
        prompt = f"""
        Create a specific, actionable task to address this system need.
        
        Need Analysis:
        {json.dumps(need, indent=2)}
        
        System Charter Guidelines:
        - Task Types: {json.dumps(charter.get('TASK_TYPES', {}), indent=2)}
        - Project Methodology: {charter.get('PROJECT_METHODOLOGY', '')}
        - Decision Principles: {json.dumps(charter.get('DECISION_PRINCIPLES', []), indent=2)}
        
        Current Projects (for FEATURE tasks):
        {json.dumps(context.get('projects', []), indent=2)}
        
        Based on the need type '{need['need_type']}', generate an appropriate task:
        
        If portfolio_expansion → Create a NEW_PROJECT task for a complete application
        If project_enhancement → Create a FEATURE task for a specific existing project
        If system_improvement → Create an improvement task for the AI system
        If quality_focus → Create appropriate testing/refactoring task
        
        CRITICAL RULES:
        1. NEW_PROJECT must describe a complete, standalone application
        2. NEW_PROJECT must explicitly mention using Laravel React starter kit
        3. FEATURE must target a specific existing project by name
        4. improvement must enhance the AI orchestrator system itself
        
        Generate the task as JSON with:
        - type: The task type (use actual enum values)
        - title: Clear, specific title (max 100 chars)
        - description: Detailed description including:
          * What needs to be built/done
          * Why it's valuable
          * Key requirements
          * Technical approach
          * For NEW_PROJECT: How to fork and customize Laravel React starter kit
        - requirements: List of specific requirements (5-8 items)
        - priority: Task priority based on need
        - estimated_complexity: 'low', 'medium', 'high'
        - success_criteria: How to measure successful completion
        - target_project: For FEATURE tasks, which project to enhance
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        task = self._parse_json_response(response)
        
        # Ensure task has all required fields
        task = self._ensure_task_completeness(task)
        
        return task
    
    def _ensure_task_completeness(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure task has all required fields.
        
        Args:
            task: Generated task
            
        Returns:
            Complete task
        """
        required_fields = ['type', 'title', 'description', 'requirements', 'priority']
        
        for field in required_fields:
            if field not in task:
                self.logger.warning(f"Task missing required field: {field}")
                # Add minimal default
                if field == 'type':
                    task['type'] = 'improvement'
                elif field == 'requirements':
                    task['requirements'] = []
                else:
                    task[field] = 'Not specified'
                    
        return task
    
    async def _ensure_starter_kit_integration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure NEW_PROJECT tasks properly use Laravel React starter kit.
        
        Args:
            task: Generated task
            
        Returns:
            Updated task
        """
        description = task.get('description', '')
        
        # Check if starter kit is mentioned
        if 'starter kit' not in description.lower() and 'laravel react' not in description.lower():
            prompt = f"""
            This NEW_PROJECT task needs to properly integrate the Laravel React starter kit.
            
            Current Task:
            {json.dumps(task, indent=2)}
            
            Update the task to:
            1. Explicitly state it will fork https://github.com/laravel/react-starter-kit
            2. Explain how to customize the starter kit for this specific project
            3. List what modifications are needed from the base template
            4. Include setup steps specific to this project type
            5. Maintain all Laravel and React best practices
            
            The starter kit provides:
            - Laravel 10+ backend with Sanctum auth
            - React 18+ with TypeScript frontend
            - Tailwind CSS styling
            - PostgreSQL database setup
            - Redis for caching
            - Docker configuration
            - CI/CD pipeline templates
            
            Return the complete updated task maintaining the same structure.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt)
            updated_task = self._parse_json_response(response)
            
            # Merge updates
            if updated_task:
                task.update(updated_task)
                
        return task
    
    async def _apply_learned_improvements(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply improvements based on learning system insights.
        
        Args:
            task: Generated task
            
        Returns:
            Improved task
        """
        if not self.learning_system:
            return task
            
        # Get recommendations
        recommendations = await self.learning_system.get_recommendations()
        
        if recommendations.get('status') == 'insufficient_data':
            return task
            
        prompt = f"""
        Improve this task based on learned insights about what creates value.
        
        Generated Task:
        {json.dumps(task, indent=2)}
        
        Learning Insights:
        - Success Patterns: {json.dumps(recommendations.get('success_template', {}), indent=2)}
        - Task Priorities: {json.dumps(recommendations.get('task_priorities', []), indent=2)}
        - Things to Avoid: {json.dumps(recommendations.get('avoid_list', []), indent=2)}
        
        Improve the task by:
        1. Incorporating successful patterns
        2. Avoiding known failure patterns
        3. Aligning with recommended priorities
        4. Making it more specific and actionable
        5. Ensuring high value creation potential
        
        Return the improved task with the same structure.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        improved_task = self._parse_json_response(response)
        
        if improved_task:
            # Keep original type and key fields
            improved_task['type'] = task['type']
            return improved_task
            
        return task
    
    async def _modify_for_higher_value(self, task: Dict[str, Any], 
                                      prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Modify task based on value prediction to increase potential value.
        
        Args:
            task: Original task
            prediction: Value prediction with improvements
            
        Returns:
            Modified task
        """
        improvements = prediction.get('value_improvements', [])
        
        if not improvements:
            return task
            
        prompt = f"""
        Modify this task to increase its potential value based on predictions.
        
        Original Task:
        {json.dumps(task, indent=2)}
        
        Predicted Issues:
        - Predicted Value: {prediction.get('predicted_value', 0)}
        - Risks: {json.dumps(prediction.get('risks', []), indent=2)}
        
        Suggested Improvements:
        {json.dumps(improvements, indent=2)}
        
        Modify the task to:
        1. Address identified risks
        2. Incorporate value improvements
        3. Increase success probability
        4. Maintain task feasibility
        
        Return the modified task with the same structure but improved content.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        modified_task = self._parse_json_response(response)
        
        if modified_task:
            # Preserve type
            modified_task['type'] = task['type']
            return modified_task
            
        return task
    
    def _record_generation(self, task: Dict[str, Any], need: Dict[str, Any]) -> None:
        """Record task generation for pattern analysis.
        
        Args:
            task: Generated task
            need: Need analysis that led to task
        """
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task': task,
            'need_analysis': need,
            'generation_id': f"gen_{len(self.task_history)}_{datetime.now(timezone.utc).timestamp()}"
        }
        
        self.task_history.append(record)
        
        # Update generation patterns
        need_type = need.get('need_type', 'unknown')
        if need_type not in self.generation_patterns:
            self.generation_patterns[need_type] = []
            
        self.generation_patterns[need_type].append({
            'task_type': task.get('type'),
            'priority': task.get('priority'),
            'timestamp': record['timestamp']
        })
    
    async def generate_multiple_tasks(self, context: Dict[str, Any], 
                                     count: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple diverse tasks.
        
        Args:
            context: System context
            count: Number of tasks to generate
            
        Returns:
            List of generated tasks
        """
        tasks = []
        used_needs = []
        
        for i in range(count):
            # Update context with already generated tasks
            generation_context = context.copy()
            generation_context['pending_tasks'] = tasks
            
            # Generate diverse tasks by avoiding same need types
            task = await self.generate_task(generation_context)
            
            # Ensure diversity
            if i > 0:
                task = await self._ensure_task_diversity(task, tasks)
                
            tasks.append(task)
            
        return tasks
    
    async def _ensure_task_diversity(self, task: Dict[str, Any], 
                                    existing_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure task is diverse from existing tasks.
        
        Args:
            task: New task
            existing_tasks: Already generated tasks
            
        Returns:
            Potentially modified task for diversity
        """
        # Check if task is too similar to existing ones
        task_types = [t.get('type') for t in existing_tasks]
        
        if task.get('type') in task_types:
            # Try to generate a different type
            prompt = f"""
            This task is too similar to already generated tasks.
            
            Current Task:
            {json.dumps(task, indent=2)}
            
            Existing Task Types: {task_types}
            
            Generate a different but equally valuable task that:
            1. Uses a different task type if possible
            2. Addresses a different aspect of the system
            3. Maintains high value potential
            4. Complements the existing tasks
            
            Return a completely different task.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt)
            diverse_task = self._parse_json_response(response)
            
            if diverse_task and diverse_task.get('type') not in task_types:
                return diverse_task
                
        return task
    
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
    
    def get_generation_analytics(self) -> Dict[str, Any]:
        """Get analytics about task generation patterns.
        
        Returns:
            Generation analytics
        """
        return {
            'total_generated': len(self.task_history),
            'generation_patterns': self.generation_patterns,
            'recent_tasks': [r['task'] for r in self.task_history[-5:]],
            'need_distribution': self._analyze_need_distribution()
        }
    
    def _analyze_need_distribution(self) -> Dict[str, int]:
        """Analyze distribution of need types.
        
        Returns:
            Need type counts
        """
        distribution = {}
        for record in self.task_history:
            need_type = record['need_analysis'].get('need_type', 'unknown')
            distribution[need_type] = distribution.get(need_type, 0) + 1
            
        return distribution