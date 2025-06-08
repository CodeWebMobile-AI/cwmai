"""
Dynamic Task Validator

Validates tasks using AI reasoning, not hardcoded rules.
Ensures tasks align with system charter and make sense.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple


class DynamicTaskValidator:
    """Validate tasks using dynamic AI-based logic."""
    
    def __init__(self, ai_brain, charter_system):
        """Initialize with AI brain and charter system.
        
        Args:
            ai_brain: AI brain for validation logic
            charter_system: Dynamic charter for guidelines
        """
        self.ai_brain = ai_brain
        self.charter_system = charter_system
        self.validation_history = []
        self.validation_patterns = {}
        self.logger = logging.getLogger(__name__)
        
    async def validate_task(self, task: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task using AI reasoning against charter and context.
        
        Args:
            task: Task to validate
            context: Current system context
            
        Returns:
            Validation result with reasoning
        """
        self.logger.info(f"Validating task: {task.get('title', 'Unknown')}")
        
        # Get current charter
        charter = await self.charter_system.get_current_charter()
        
        # First, check charter alignment
        charter_validation = await self.charter_system.validate_against_charter(task)
        
        if not charter_validation.get('aligned', True):
            return {
                'valid': False,
                'issues': ['Task does not align with system charter'],
                'charter_reasoning': charter_validation.get('reasoning', ''),
                'suggestions': charter_validation.get('suggestions', [])
            }
        
        # Perform comprehensive validation
        validation_result = await self._comprehensive_validation(task, context, charter)
        
        # Record for learning
        self._record_validation(task, validation_result, context)
        
        # If invalid but fixable, generate corrected version
        if not validation_result['valid'] and validation_result.get('fixable', False):
            corrected_task = await self.suggest_task_correction(
                task, 
                validation_result['issues']
            )
            validation_result['corrected_task'] = corrected_task
            
        return validation_result
    
    async def _comprehensive_validation(self, task: Dict[str, Any],
                                       context: Dict[str, Any],
                                       charter: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive task validation.
        
        Args:
            task: Task to validate
            context: System context
            charter: System charter
            
        Returns:
            Detailed validation result
        """
        prompt = f"""
        Perform comprehensive validation of this task for an AI development orchestrator.
        
        Task to Validate:
        {json.dumps(task, indent=2)}
        
        System Charter:
        {json.dumps(charter, indent=2)}
        
        Current Context:
        - Active Projects: {json.dumps(context.get('projects', []), indent=2)}
        - Recent Tasks: {json.dumps(context.get('recent_tasks', [])[-5:], indent=2)}
        - Pending Tasks: {json.dumps(context.get('pending_tasks', []), indent=2)}
        
        Validation Criteria:
        
        1. ALIGNMENT: Does this task align with the system's purpose and objectives?
        
        2. TYPE APPROPRIATENESS:
           - NEW_PROJECT: Is this a complete, standalone application?
           - FEATURE: Does it target a real, existing project?
           - improvement: Does it enhance the AI system itself?
           
        3. FEASIBILITY: Can this be realistically completed?
        
        4. VALUE: Will this create meaningful value?
        
        5. SPECIFICITY: Is the task specific enough to execute?
        
        6. DUPLICATION: Is this duplicating existing work?
        
        7. DEPENDENCIES: Are dependencies properly identified?
        
        8. LARAVEL REACT: For NEW_PROJECT, does it mention the starter kit?
        
        Common Issues to Check:
        - FEATURE tasks targeting non-existent projects
        - Vague or ambiguous requirements
        - Busy work that doesn't advance the mission
        - Tasks that are too large or should be split
        - Missing technical specifications
        
        Return validation as JSON with:
        - valid: true/false
        - score: 0.0 to 1.0 (overall quality score)
        - issues: List of specific problems found
        - strengths: Positive aspects of the task
        - fixable: Whether issues can be fixed
        - suggestions: Specific fixes for each issue
        - reasoning: Detailed explanation of validation
        - risk_level: 'low', 'medium', 'high'
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_validation_response(response)
    
    async def suggest_task_correction(self, task: Dict[str, Any], 
                                     issues: List[str]) -> Dict[str, Any]:
        """Get AI suggestion for fixing invalid task.
        
        Args:
            task: Invalid task
            issues: List of validation issues
            
        Returns:
            Corrected task
        """
        self.logger.info(f"Generating correction for task with {len(issues)} issues")
        
        # Get charter for guidelines
        charter = await self.charter_system.get_current_charter()
        
        prompt = f"""
        Fix this invalid task based on identified issues.
        
        Invalid Task:
        {json.dumps(task, indent=2)}
        
        Validation Issues:
        {json.dumps(issues, indent=2)}
        
        System Charter Guidelines:
        - Task Types: {json.dumps(charter.get('TASK_TYPES', {}), indent=2)}
        - Project Methodology: {charter.get('PROJECT_METHODOLOGY', '')}
        - Primary Purpose: {charter.get('PRIMARY_PURPOSE', '')}
        
        Common Corrections:
        1. If FEATURE without project → Convert to NEW_PROJECT for a complete app
        2. If too vague → Add specific requirements and success criteria
        3. If busy work → Reframe to create real value
        4. If missing starter kit → Add Laravel React starter kit integration
        5. If too large → Split into focused, achievable task
        
        Generate a corrected version that:
        - Addresses all identified issues
        - Maintains the original intent where valid
        - Aligns with system charter
        - Is specific and actionable
        - Creates clear value
        
        Return the complete corrected task with same structure as original.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        corrected = self._parse_json_response(response)
        
        # Ensure corrected task has all required fields
        if corrected:
            # Preserve certain fields from original if not updated
            for field in ['type', 'priority']:
                if field not in corrected and field in task:
                    corrected[field] = task[field]
                    
        return corrected
    
    async def validate_batch(self, tasks: List[Dict[str, Any]], 
                           context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate multiple tasks, considering their relationships.
        
        Args:
            tasks: List of tasks to validate
            context: System context
            
        Returns:
            List of validation results
        """
        validations = []
        
        # Validate each task with awareness of others
        for i, task in enumerate(tasks):
            # Add other tasks to context
            batch_context = context.copy()
            batch_context['batch_tasks'] = [t for j, t in enumerate(tasks) if j != i]
            
            validation = await self.validate_task(task, batch_context)
            
            # Check for batch-specific issues
            batch_issues = await self._check_batch_issues(task, tasks, i)
            if batch_issues:
                validation['batch_issues'] = batch_issues
                if batch_issues.get('critical'):
                    validation['valid'] = False
                    validation['issues'].extend(batch_issues['issues'])
                    
            validations.append(validation)
            
        return validations
    
    async def _check_batch_issues(self, task: Dict[str, Any], 
                                 all_tasks: List[Dict[str, Any]], 
                                 index: int) -> Optional[Dict[str, Any]]:
        """Check for issues specific to task batches.
        
        Args:
            task: Current task
            all_tasks: All tasks in batch
            index: Index of current task
            
        Returns:
            Batch-specific issues if any
        """
        other_tasks = [t for i, t in enumerate(all_tasks) if i != index]
        
        prompt = f"""
        Check for batch-specific issues with this task.
        
        Current Task:
        {json.dumps(task, indent=2)}
        
        Other Tasks in Batch:
        {json.dumps(other_tasks, indent=2)}
        
        Check for:
        1. DUPLICATION: Is this task too similar to another?
        2. CONFLICTS: Do tasks conflict with each other?
        3. DEPENDENCIES: Should tasks be ordered differently?
        4. RESOURCE COMPETITION: Do tasks compete for same resources?
        5. BALANCE: Is the batch well-balanced across types/priorities?
        
        Return findings as JSON with:
        - has_issues: true/false
        - critical: true/false (if issues prevent execution)
        - issues: List of specific batch issues
        - suggestions: How to resolve the issues
        - optimal_order: Suggested task execution order if relevant
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        batch_analysis = self._parse_json_response(response)
        
        if batch_analysis.get('has_issues'):
            return batch_analysis
            
        return None
    
    def _record_validation(self, task: Dict[str, Any], 
                          validation: Dict[str, Any],
                          context: Dict[str, Any]) -> None:
        """Record validation for pattern learning.
        
        Args:
            task: Validated task
            validation: Validation result
            context: Context at validation time
        """
        record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task': task,
            'validation': validation,
            'context_snapshot': {
                'project_count': len(context.get('projects', [])),
                'pending_tasks': len(context.get('pending_tasks', []))
            }
        }
        
        self.validation_history.append(record)
        
        # Update validation patterns
        task_type = task.get('type', 'unknown')
        if task_type not in self.validation_patterns:
            self.validation_patterns[task_type] = {
                'total': 0,
                'valid': 0,
                'common_issues': {},
                'success_patterns': []
            }
            
        pattern = self.validation_patterns[task_type]
        pattern['total'] += 1
        
        if validation['valid']:
            pattern['valid'] += 1
            # Record what made this task valid
            if validation.get('strengths'):
                pattern['success_patterns'].extend(validation['strengths'])
        else:
            # Track common issues
            for issue in validation.get('issues', []):
                pattern['common_issues'][issue] = pattern['common_issues'].get(issue, 0) + 1
                
        # Learn from patterns every 20 validations
        if len(self.validation_history) % 20 == 0:
            asyncio.create_task(self._analyze_validation_patterns())
    
    async def _analyze_validation_patterns(self) -> None:
        """Analyze validation patterns to improve future validations."""
        prompt = f"""
        Analyze validation patterns to identify improvements.
        
        Validation Patterns:
        {json.dumps(self.validation_patterns, indent=2)}
        
        Recent Validation History (last 10):
        {json.dumps(self.validation_history[-10:], indent=2)}
        
        Identify:
        1. Most common validation failures by task type
        2. Patterns that lead to valid tasks
        3. Systemic issues in task generation
        4. Recommendations for task generator
        5. Validation criteria that might need adjustment
        
        Return analysis as JSON with:
        - key_findings: Main discoveries from patterns
        - generation_improvements: How task generation can improve
        - validation_adjustments: Any validation criteria to adjust
        - success_template: What makes a high-quality task
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        analysis = self._parse_json_response(response)
        
        # Store insights
        if analysis:
            self.validation_patterns['_analysis'] = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'insights': analysis
            }
    
    async def get_validation_insights(self) -> Dict[str, Any]:
        """Get insights from validation history.
        
        Returns:
            Validation insights and recommendations
        """
        if len(self.validation_history) < 5:
            return {'status': 'insufficient_data'}
            
        # Calculate metrics
        total_validations = len(self.validation_history)
        valid_count = sum(1 for r in self.validation_history if r['validation']['valid'])
        validity_rate = valid_count / total_validations
        
        # Get pattern analysis
        pattern_analysis = self.validation_patterns.get('_analysis', {})
        
        return {
            'total_validations': total_validations,
            'validity_rate': validity_rate,
            'patterns_by_type': {
                k: v for k, v in self.validation_patterns.items() 
                if k != '_analysis'
            },
            'latest_analysis': pattern_analysis.get('insights', {}),
            'recommendations': await self._generate_recommendations()
        }
    
    async def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations based on validation patterns.
        
        Returns:
            Recommendations for improvement
        """
        prompt = f"""
        Generate actionable recommendations based on validation patterns.
        
        Validation Statistics:
        {json.dumps({
            'total': len(self.validation_history),
            'patterns': self.validation_patterns
        }, indent=2)}
        
        Generate recommendations for:
        1. Task generator improvements
        2. Common issues to avoid
        3. Task quality guidelines
        4. Validation process improvements
        
        Be specific and actionable.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        return self._parse_json_response(response)
    
    def _parse_validation_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse validation response from AI.
        
        Args:
            response: AI response
            
        Returns:
            Parsed validation result
        """
        parsed = self._parse_json_response(response)
        
        # Ensure required fields
        if 'valid' not in parsed:
            parsed['valid'] = True  # Default to valid if not specified
            
        if 'issues' not in parsed:
            parsed['issues'] = []
            
        if 'score' not in parsed:
            parsed['score'] = 0.8 if parsed['valid'] else 0.3
            
        return parsed
    
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
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation activities.
        
        Returns:
            Validation summary
        """
        if not self.validation_history:
            return {'status': 'no_validations_yet'}
            
        recent_validations = self.validation_history[-20:]
        
        return {
            'total_validations': len(self.validation_history),
            'recent_validity_rate': sum(
                1 for r in recent_validations if r['validation']['valid']
            ) / len(recent_validations),
            'common_issues': self._get_common_issues(),
            'validation_patterns': self.validation_patterns
        }
    
    def _get_common_issues(self) -> List[Tuple[str, int]]:
        """Get most common validation issues.
        
        Returns:
            List of (issue, count) tuples
        """
        issue_counts = {}
        
        for record in self.validation_history:
            for issue in record['validation'].get('issues', []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
        # Sort by frequency
        sorted_issues = sorted(
            issue_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_issues[:5]  # Top 5 issues