"""Alternative Task Generator for handling duplicate task scenarios.

This module generates intelligent alternative tasks when duplicates are detected,
ensuring workers remain productive instead of sitting idle.
"""

import asyncio
import logging
import random
import json
import re
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from scripts.work_item_types import WorkItem, TaskPriority


class AlternativeTaskGenerator:
    """Generates alternative tasks when duplicates are encountered."""
    
    def __init__(self, ai_brain=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.ai_brain = ai_brain
        
        # Task variation templates
        self.task_variations = {
            "documentation": [
                "Create API examples for {feature}",
                "Write user guide for {feature}",
                "Document edge cases for {feature}",
                "Create troubleshooting guide for {feature}",
                "Write migration guide for {feature}",
                "Document configuration options for {feature}",
                "Create FAQ section for {feature}"
            ],
            "testing": [
                "Write unit tests for {feature}",
                "Create integration tests for {feature}",
                "Add edge case tests for {feature}",
                "Implement performance tests for {feature}",
                "Create end-to-end tests for {feature}",
                "Add security tests for {feature}",
                "Write regression tests for {feature}"
            ],
            "optimization": [
                "Optimize performance of {feature}",
                "Reduce memory usage in {feature}",
                "Improve caching for {feature}",
                "Optimize database queries in {feature}",
                "Reduce code complexity in {feature}",
                "Improve error handling in {feature}",
                "Optimize API response times for {feature}"
            ],
            "refactoring": [
                "Extract reusable components from {feature}",
                "Improve code structure in {feature}",
                "Apply design patterns to {feature}",
                "Split large functions in {feature}",
                "Remove code duplication in {feature}",
                "Improve naming conventions in {feature}",
                "Modernize legacy code in {feature}"
            ],
            "feature_enhancement": [
                "Add logging to {feature}",
                "Implement analytics for {feature}",
                "Add configuration options to {feature}",
                "Improve user feedback in {feature}",
                "Add accessibility features to {feature}",
                "Implement caching for {feature}",
                "Add monitoring capabilities to {feature}"
            ],
            "maintenance": [
                "Update dependencies for {repository}",
                "Clean up deprecated code in {repository}",
                "Fix linting issues in {repository}",
                "Update README for {repository}",
                "Organize project structure in {repository}",
                "Add CI/CD improvements to {repository}",
                "Review and close stale issues in {repository}"
            ]
        }
        
        # Task type mapping for alternatives
        self.alternative_task_types = {
            "DOCUMENTATION": ["TESTING", "FEATURE", "MAINTENANCE"],
            "TESTING": ["DOCUMENTATION", "OPTIMIZATION", "REFACTORING"],
            "FEATURE": ["TESTING", "DOCUMENTATION", "OPTIMIZATION"],
            "BUG_FIX": ["TESTING", "DOCUMENTATION", "REFACTORING"],
            "OPTIMIZATION": ["TESTING", "MONITORING", "DOCUMENTATION"],
            "REFACTORING": ["TESTING", "DOCUMENTATION", "OPTIMIZATION"],
            "MAINTENANCE": ["DOCUMENTATION", "TESTING", "MONITORING"]
        }
    
    async def generate_alternative_task(self, 
                                      original_task: WorkItem,
                                      context: Optional[Dict[str, Any]] = None) -> Optional[WorkItem]:
        """Generate an alternative task when a duplicate is detected.
        
        Args:
            original_task: The task that was detected as duplicate
            context: Additional context about the project/repository
            
        Returns:
            Alternative WorkItem or None if generation fails
        """
        try:
            context = context or {}
            attempted_alternatives = context.get('attempted_alternatives', [])
            attempt_number = context.get('attempt_number', 1)
            
            self.logger.info(
                f"Generating alternative for duplicate task: {original_task.title} "
                f"(attempt {attempt_number}, {len(attempted_alternatives)} already tried)"
            )
            
            # Try AI-based generation first if available
            if self.ai_brain:
                ai_alternative = await self._generate_ai_alternative(original_task, context)
                if ai_alternative:
                    # Check if this alternative has already been attempted
                    if ai_alternative.title.lower() not in [alt.lower() for alt in attempted_alternatives]:
                        return ai_alternative
                    else:
                        self.logger.debug(f"AI generated already-attempted alternative: {ai_alternative.title}")
            
            # Fallback to template-based generation
            template_alternative = await self._generate_template_alternative(original_task, context)
            
            # Ensure template alternative is not in attempted list
            if template_alternative and template_alternative.title.lower() not in [alt.lower() for alt in attempted_alternatives]:
                return template_alternative
            
            # If all else fails, try to generate a completely different task type
            if attempt_number >= 2:
                return await self._generate_different_type_task(original_task, context)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to generate alternative task: {e}")
            return None
    
    async def _generate_ai_alternative(self, 
                                     original_task: WorkItem,
                                     context: Optional[Dict[str, Any]]) -> Optional[WorkItem]:
        """Use AI to generate a contextually relevant alternative task."""
        if not self.ai_brain:
            return None
        
        try:
            context = context or {}
            attempted_alternatives = context.get('attempted_alternatives', [])
            attempt_number = context.get('attempt_number', 1)
            
            # Build a list of what NOT to generate
            avoid_list = [original_task.title] + attempted_alternatives
            
            prompt = f"""A worker tried to execute this task but it was already completed:
Task: {original_task.title}
Type: {original_task.task_type}
Repository: {original_task.repository}
Description: {original_task.description[:200]}...

This is attempt #{attempt_number} to find an alternative task.

Already attempted alternatives that were also duplicates:
{json.dumps(attempted_alternatives, indent=2) if attempted_alternatives else "None"}

Generate a COMPLETELY DIFFERENT but valuable task for the same repository.
The task must be:
1. Substantially different from the original and all attempted alternatives
2. Add real value to the project
3. Be actionable and specific
4. NOT just a variation of the same task

Consider these alternative approaches:
- If original was about features, suggest testing or documentation
- If original was about code, suggest infrastructure or tooling
- If original was about one component, suggest a different component
- If original was about implementation, suggest planning or design

IMPORTANT: The title must be unique and NOT similar to:
{json.dumps(avoid_list, indent=2)}

Provide a new task with:
- title: Brief, unique task title
- description: What needs to be done
- task_type: One of FEATURE, TESTING, DOCUMENTATION, OPTIMIZATION, REFACTORING, MAINTENANCE
- priority: HIGH, MEDIUM, or LOW
- estimated_cycles: 1-5

Format as JSON."""

            response = await self.ai_brain.generate_enhanced_response(prompt)
            
            # Parse AI response and create WorkItem
            # The response is a dict with 'content' field containing the JSON string
            content = response.get('content', '{}')
            
            # Handle empty or non-JSON content
            if not content or content.strip() == '':
                self.logger.warning("AI returned empty content for alternative task generation")
                return None
                
            try:
                # Extract JSON from various formats
                extracted_json = self._extract_json_from_response(content)
                if not extracted_json:
                    self.logger.warning(f"Could not extract JSON from AI response. Content: {content[:200]}...")
                    return None
                
                task_data = json.loads(extracted_json)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse AI response as JSON: {e}")
                self.logger.debug(f"Original content: {content[:500]}...")
                self.logger.debug(f"Extracted content: {extracted_json[:500] if extracted_json else 'None'}...")
                return None
            
            return WorkItem(
                id=f"alt_{original_task.id}_{datetime.now(timezone.utc).timestamp()}",
                task_type=task_data.get('task_type', 'FEATURE'),
                title=task_data.get('title', 'Alternative task'),
                description=task_data.get('description', ''),
                priority=TaskPriority[task_data.get('priority', 'MEDIUM')],
                repository=original_task.repository,
                estimated_cycles=task_data.get('estimated_cycles', 2),
                metadata={
                    'alternative_for': original_task.id,
                    'generation_method': 'ai'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"AI alternative generation failed: {e}")
            return None
    
    async def _generate_template_alternative(self,
                                           original_task: WorkItem,
                                           context: Optional[Dict[str, Any]]) -> WorkItem:
        """Generate alternative using predefined templates."""
        # Extract feature/component name from original task
        feature_name = self._extract_feature_name(original_task.title)
        
        # Determine alternative task type
        original_type = original_task.task_type
        alternative_types = self.alternative_task_types.get(
            original_type, 
            ["TESTING", "DOCUMENTATION", "OPTIMIZATION"]
        )
        new_task_type = random.choice(alternative_types)
        
        # Select appropriate template category
        template_category = self._get_template_category(new_task_type)
        templates = self.task_variations.get(template_category, [])
        
        if not templates:
            # Generic fallback
            title = f"Review and improve {feature_name}"
            description = f"Review the implementation of {feature_name} and identify improvement opportunities"
        else:
            # Use template
            template = random.choice(templates)
            title = template.format(
                feature=feature_name,
                repository=original_task.repository or "the project"
            )
            description = f"Alternative task generated because '{original_task.title}' was already completed. {title}"
        
        # Create alternative work item
        return WorkItem(
            id=f"alt_{original_task.id}_{datetime.now(timezone.utc).timestamp()}",
            task_type=new_task_type,
            title=title,
            description=description,
            priority=TaskPriority.MEDIUM,  # Generally lower priority than originals
            repository=original_task.repository,
            estimated_cycles=max(1, original_task.estimated_cycles - 1),
            metadata={
                'alternative_for': original_task.id,
                'generation_method': 'template',
                'original_title': original_task.title
            }
        )
    
    async def _generate_different_type_task(self, 
                                          original_task: WorkItem,
                                          context: Optional[Dict[str, Any]]) -> Optional[WorkItem]:
        """Generate a completely different type of task as last resort."""
        try:
            # Define task types that are always valuable
            safe_alternatives = {
                'DOCUMENTATION': {
                    'title': f"Update README for {original_task.repository or 'project'}",
                    'description': "Review and update the README file with current project status, features, and setup instructions"
                },
                'TESTING': {
                    'title': f"Add test coverage report for {original_task.repository or 'project'}",
                    'description': "Set up test coverage reporting and identify areas needing more tests"
                },
                'MAINTENANCE': {
                    'title': f"Clean up old issues in {original_task.repository or 'project'}",
                    'description': "Review and close stale issues, update labels, and organize the issue tracker"
                },
                'OPTIMIZATION': {
                    'title': f"Profile performance bottlenecks in {original_task.repository or 'project'}",
                    'description': "Run performance profiling to identify and document optimization opportunities"
                }
            }
            
            # Pick a type different from the original
            available_types = [t for t in safe_alternatives.keys() if t != original_task.task_type]
            if not available_types:
                available_types = list(safe_alternatives.keys())
            
            chosen_type = random.choice(available_types)
            task_info = safe_alternatives[chosen_type]
            
            return WorkItem(
                id=f"alt_fallback_{uuid.uuid4().hex[:8]}",
                task_type=chosen_type,
                title=task_info['title'],
                description=f"{task_info['description']}. (Generated as safe alternative after multiple duplicate attempts)",
                priority=TaskPriority.LOW,
                repository=original_task.repository,
                estimated_cycles=2,
                metadata={
                    'alternative_for': original_task.id,
                    'generation_method': 'safe_fallback',
                    'original_title': original_task.title,
                    'fallback_reason': 'exhausted_alternatives'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate different type task: {e}")
            return None
    
    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """Extract JSON from various AI response formats."""
        if not content:
            return None
            
        content = content.strip()
        
        # Method 1: Check if the entire content is valid JSON
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            pass
        
        # Method 2: Strip markdown code blocks
        if '```' in content:
            # Match content between ```json and ``` or just between ``` and ```
            patterns = [
                r'```json\s*\n(.*?)\n```',
                r'```\s*\n(.*?)\n```'
            ]
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    extracted = match.group(1).strip()
                    try:
                        json.loads(extracted)
                        return extracted
                    except json.JSONDecodeError:
                        continue
        
        # Method 3: Find JSON object using regex
        # Look for content between { and }
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
        
        # Method 4: Try to extract JSON after common prefixes
        prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "JSON:",
            "Response:",
            "Output:"
        ]
        
        for prefix in prefixes:
            if prefix in content:
                parts = content.split(prefix, 1)
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    # Try to parse the remaining content
                    try:
                        json.loads(remaining)
                        return remaining
                    except json.JSONDecodeError:
                        # Try to find JSON in the remaining content
                        for match in re.findall(json_pattern, remaining, re.DOTALL):
                            try:
                                json.loads(match)
                                return match
                            except json.JSONDecodeError:
                                continue
        
        return None
    
    def _extract_feature_name(self, task_title: str) -> str:
        """Extract the feature/component name from a task title."""
        # Remove common task prefixes
        prefixes = [
            "Update documentation for", "Document", "Test", "Fix",
            "Implement", "Add", "Create", "Write tests for",
            "Optimize", "Refactor", "Review", "Improve"
        ]
        
        feature = task_title
        for prefix in prefixes:
            if feature.lower().startswith(prefix.lower()):
                feature = feature[len(prefix):].strip()
                break
        
        # Remove common suffixes
        suffixes = ["changes", "updates", "improvements", "feature", "component"]
        for suffix in suffixes:
            if feature.lower().endswith(suffix.lower()):
                feature = feature[:-len(suffix)].strip()
        
        return feature or "the feature"
    
    def _get_template_category(self, task_type: str) -> str:
        """Map task type to template category."""
        mapping = {
            "DOCUMENTATION": "documentation",
            "TESTING": "testing",
            "OPTIMIZATION": "optimization",
            "REFACTORING": "refactoring",
            "FEATURE": "feature_enhancement",
            "MAINTENANCE": "maintenance",
            "BUG_FIX": "testing",
            "MONITORING": "feature_enhancement"
        }
        return mapping.get(task_type, "feature_enhancement")
    
    async def generate_alternative_batch(self,
                                       original_tasks: List[WorkItem],
                                       max_alternatives: int = 5) -> List[WorkItem]:
        """Generate multiple alternative tasks for a batch of duplicates.
        
        This is useful when multiple workers hit duplicates simultaneously.
        """
        alternatives = []
        seen_titles = set()
        
        for task in original_tasks[:max_alternatives]:
            alternative = await self.generate_alternative_task(task)
            if alternative and alternative.title not in seen_titles:
                alternatives.append(alternative)
                seen_titles.add(alternative.title)
        
        return alternatives