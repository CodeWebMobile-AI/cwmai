"""
Progressive Task Generator

Dynamically generates follow-up tasks based on completion patterns, learning from outcomes,
and intelligent prediction of next steps in development workflows.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

from ai_brain import IntelligentAIBrain
from hierarchical_task_manager import HierarchicalTaskManager, TaskNode
from complexity_analyzer import ComplexityAnalyzer


class ProgressionTrigger(Enum):
    """Triggers for progressive task generation."""
    TASK_COMPLETION = "task_completion"
    MILESTONE_REACHED = "milestone_reached"
    BLOCKER_RESOLVED = "blocker_resolved"
    PATTERN_DETECTED = "pattern_detected"
    TIME_BASED = "time_based"
    REPOSITORY_CHANGE = "repository_change"
    EXTERNAL_EVENT = "external_event"


class TaskRelationship(Enum):
    """Types of relationships between tasks."""
    SEQUENTIAL = "sequential"       # Next logical step
    DEPENDENCY = "dependency"       # Required prerequisite
    ENHANCEMENT = "enhancement"     # Builds upon completed work
    MAINTENANCE = "maintenance"     # Follow-up maintenance
    EXPLORATION = "exploration"     # Investigative next step
    OPTIMIZATION = "optimization"   # Performance/quality improvement


@dataclass
class ProgressionPattern:
    """Pattern for task progression."""
    name: str
    trigger: ProgressionTrigger
    conditions: List[str]
    next_task_types: List[str]
    relationship_type: TaskRelationship
    confidence: float
    success_rate: float
    usage_count: int = 0


@dataclass
class NextTaskSuggestion:
    """Suggestion for next task to generate."""
    title: str
    description: str
    task_type: str
    priority: str
    estimated_hours: float
    relationship: TaskRelationship
    trigger_reason: str
    confidence: float
    prerequisites: List[str]
    context_factors: List[str]
    generated_from: str  # Parent task ID


@dataclass
class ProgressionContext:
    """Context for progressive task generation."""
    completed_task: Dict[str, Any]
    repository_context: Dict[str, Any]
    project_state: Dict[str, Any]
    recent_patterns: List[str]
    current_priorities: List[str]
    ai_agent_capacity: Dict[str, Any]  # AI system capacity, not human team
    processing_constraints: Dict[str, Any]  # AI processing constraints, not timeline


class ProgressiveTaskGenerator:
    """Generates follow-up tasks based on completion patterns and intelligent prediction."""
    
    def __init__(self, ai_brain: IntelligentAIBrain, 
                 hierarchical_manager: HierarchicalTaskManager,
                 complexity_analyzer: ComplexityAnalyzer,
                 context_aggregator=None,
                 predictive_engine=None):
        """Initialize the progressive task generator.
        
        Args:
            ai_brain: AI brain for intelligent analysis
            hierarchical_manager: Hierarchical task manager
            complexity_analyzer: Complexity analyzer
            context_aggregator: Smart context aggregator for enhanced awareness
            predictive_engine: Predictive engine for ML-based insights
        """
        self.ai_brain = ai_brain
        self.hierarchical_manager = hierarchical_manager
        self.complexity_analyzer = complexity_analyzer
        self.context_aggregator = context_aggregator
        self.predictive_engine = predictive_engine
        self.logger = logging.getLogger(__name__)
        
        # Pattern learning and storage
        self.progression_patterns: Dict[str, ProgressionPattern] = {}
        self.completion_history: List[Dict[str, Any]] = []
        self.pattern_success_tracking: Dict[str, List[bool]] = defaultdict(list)
        self.cross_project_patterns: Dict[str, List[ProgressionPattern]] = defaultdict(list)
        
        # Configuration
        self.max_suggestions_per_completion = 3
        self.min_confidence_threshold = 0.6
        self.pattern_learning_enabled = True
        self.dynamic_pattern_learning = True
        self.cross_project_recognition = True
        
        # Success tracking
        self.pattern_performance_history = defaultdict(list)
        self.adaptive_confidence_adjustment = True
        
        # Load existing patterns
        self._load_progression_patterns()
        
    async def generate_next_tasks(self, completed_task: Dict[str, Any], 
                                context: ProgressionContext) -> List[NextTaskSuggestion]:
        """Generate next task suggestions based on completed task with enhanced intelligence.
        
        Args:
            completed_task: Recently completed task
            context: Progression context
            
        Returns:
            List of next task suggestions
        """
        self.logger.info(f"Generating next tasks for completed: {completed_task.get('title', 'Unknown')}")
        
        suggestions = []
        
        # Record completion for pattern learning
        self._record_completion(completed_task, context)
        
        # Enhance context with aggregated data if available
        enhanced_context = await self._enhance_context(context)
        
        # Generate suggestions from different sources
        pattern_suggestions = await self._generate_from_patterns(completed_task, enhanced_context)
        ai_suggestions = await self._generate_from_ai_analysis(completed_task, enhanced_context)
        logical_suggestions = self._generate_logical_next_steps(completed_task, enhanced_context)
        
        # Add cross-project pattern suggestions if enabled
        if self.cross_project_recognition:
            cross_suggestions = await self._generate_from_cross_project_patterns(
                completed_task, enhanced_context
            )
            all_suggestions = pattern_suggestions + ai_suggestions + logical_suggestions + cross_suggestions
        else:
            all_suggestions = pattern_suggestions + ai_suggestions + logical_suggestions
        
        # Apply predictive insights if available
        if self.predictive_engine:
            all_suggestions = await self._enhance_with_predictions(all_suggestions, enhanced_context)
        
        # Combine and deduplicate suggestions
        unique_suggestions = self._deduplicate_suggestions(all_suggestions)
        
        # Dynamic confidence adjustment based on recent performance
        if self.adaptive_confidence_adjustment:
            unique_suggestions = self._adjust_confidence_dynamically(unique_suggestions)
        
        # Filter by confidence and relevance
        filtered_suggestions = [
            s for s in unique_suggestions 
            if s.confidence >= self.min_confidence_threshold
        ]
        
        # Sort by confidence and limit results
        filtered_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        suggestions = filtered_suggestions[:self.max_suggestions_per_completion]
        
        # Learn from generated suggestions
        if self.pattern_learning_enabled:
            await self._update_pattern_learning(completed_task, suggestions, enhanced_context)
        
        # Track success patterns if dynamic learning is enabled
        if self.dynamic_pattern_learning:
            self._track_pattern_generation(completed_task, suggestions)
        
        self.logger.info(f"Generated {len(suggestions)} next task suggestions")
        return suggestions
    
    async def _generate_from_patterns(self, completed_task: Dict[str, Any], 
                                    context: ProgressionContext) -> List[NextTaskSuggestion]:
        """Generate suggestions based on learned patterns.
        
        Args:
            completed_task: Completed task
            context: Progression context
            
        Returns:
            Pattern-based suggestions
        """
        suggestions = []
        task_type = completed_task.get('type', '').upper()
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.progression_patterns.values():
            if self._pattern_matches_context(pattern, completed_task, context):
                matching_patterns.append(pattern)
        
        # Sort patterns by success rate and confidence
        matching_patterns.sort(key=lambda p: p.success_rate * p.confidence, reverse=True)
        
        # Generate suggestions from top patterns
        for pattern in matching_patterns[:2]:  # Use top 2 patterns
            for next_task_type in pattern.next_task_types:
                suggestion = await self._create_suggestion_from_pattern(
                    pattern, next_task_type, completed_task, context
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_from_ai_analysis(self, completed_task: Dict[str, Any], 
                                       context: ProgressionContext) -> List[NextTaskSuggestion]:
        """Generate suggestions using AI analysis.
        
        Args:
            completed_task: Completed task
            context: Progression context
            
        Returns:
            AI-generated suggestions
        """
        suggestions = []
        
        try:
            prompt = f"""
            Analyze this completed task and suggest 2-3 logical next tasks:
            
            Completed Task:
            - Title: {completed_task.get('title', '')}
            - Type: {completed_task.get('type', '')}
            - Description: {completed_task.get('description', '')}
            - Repository: {completed_task.get('repository', '')}
            
            Context:
            - Project State: {json.dumps(context.project_state, indent=2)}
            - Repository Context: {json.dumps(context.repository_context, indent=2)}
            - Current Priorities: {context.current_priorities}
            
            Consider:
            1. What are the natural next steps after this completion?
            2. What dependencies might now be unblocked?
            3. What improvements or enhancements could be made?
            4. What maintenance or follow-up tasks are needed?
            5. What testing or validation should happen next?
            
            For each suggestion, provide:
            - title: Clear, actionable title
            - description: Detailed description
            - task_type: Type of task (FEATURE, BUG_FIX, TESTING, etc.)
            - priority: Priority level (low, medium, high, critical)
            - estimated_hours: Estimated hours (0.5-8 hours)
            - relationship: How it relates to completed task (sequential, enhancement, maintenance, etc.)
            - reasoning: Why this task should be done next
            
            Return as JSON array of suggestions.
            """
            
            response = await self.ai_brain.generate_enhanced_response(prompt)
            content = response.get('content', '')
            
            # Parse AI response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                ai_suggestions_data = json.loads(json_match.group())
                
                for data in ai_suggestions_data:
                    suggestion = NextTaskSuggestion(
                        title=data.get('title', 'AI-suggested task'),
                        description=data.get('description', ''),
                        task_type=data.get('task_type', 'TASK'),
                        priority=data.get('priority', 'medium'),
                        estimated_hours=float(data.get('estimated_hours', 2.0)),
                        relationship=TaskRelationship(data.get('relationship', 'sequential')),
                        trigger_reason="AI analysis of completion patterns",
                        confidence=0.7,  # AI suggestions get moderate confidence
                        prerequisites=[],
                        context_factors=['ai_analysis'],
                        generated_from=completed_task.get('id', 'unknown')
                    )
                    suggestions.append(suggestion)
                    
        except Exception as e:
            self.logger.warning(f"AI suggestion generation failed: {e}")
        
        return suggestions
    
    def _generate_logical_next_steps(self, completed_task: Dict[str, Any], 
                                   context: ProgressionContext) -> List[NextTaskSuggestion]:
        """Generate logical next steps based on task type and context.
        
        Args:
            completed_task: Completed task
            context: Progression context
            
        Returns:
            Logically derived suggestions
        """
        suggestions = []
        task_type = completed_task.get('type', '').upper()
        
        # Define logical progressions by task type
        logical_progressions = {
            'NEW_PROJECT': [
                ('Add comprehensive testing', 'TESTING', 'high', 4.0, TaskRelationship.SEQUENTIAL),
                ('Create deployment pipeline', 'INFRASTRUCTURE', 'medium', 3.0, TaskRelationship.SEQUENTIAL),
                ('Add monitoring and analytics', 'FEATURE', 'medium', 2.0, TaskRelationship.ENHANCEMENT)
            ],
            'FEATURE': [
                ('Add unit tests for new feature', 'TESTING', 'high', 2.0, TaskRelationship.SEQUENTIAL),
                ('Update documentation', 'DOCUMENTATION', 'medium', 1.0, TaskRelationship.MAINTENANCE),
                ('Performance optimization', 'PERFORMANCE', 'low', 3.0, TaskRelationship.OPTIMIZATION)
            ],
            'BUG_FIX': [
                ('Add regression tests', 'TESTING', 'high', 1.5, TaskRelationship.SEQUENTIAL),
                ('Review similar code patterns', 'CODE_REVIEW', 'medium', 2.0, TaskRelationship.EXPLORATION),
                ('Update error handling', 'REFACTOR', 'low', 2.0, TaskRelationship.ENHANCEMENT)
            ],
            'TESTING': [
                ('Analyze test coverage gaps', 'TESTING', 'medium', 1.0, TaskRelationship.EXPLORATION),
                ('Refactor based on test insights', 'REFACTOR', 'low', 3.0, TaskRelationship.OPTIMIZATION),
                ('Add integration tests', 'TESTING', 'medium', 2.0, TaskRelationship.ENHANCEMENT)
            ],
            'REFACTOR': [
                ('Update related documentation', 'DOCUMENTATION', 'medium', 1.0, TaskRelationship.MAINTENANCE),
                ('Performance benchmarking', 'PERFORMANCE', 'low', 1.5, TaskRelationship.OPTIMIZATION),
                ('Code review of changes', 'CODE_REVIEW', 'high', 1.0, TaskRelationship.SEQUENTIAL)
            ]
        }
        
        if task_type in logical_progressions:
            for title_template, next_type, priority, hours, relationship in logical_progressions[task_type]:
                # Customize title based on completed task
                customized_title = title_template
                if 'feature' in title_template.lower():
                    feature_name = completed_task.get('title', 'feature').replace('Add ', '').replace('Implement ', '')
                    customized_title = title_template.replace('new feature', feature_name)
                
                suggestion = NextTaskSuggestion(
                    title=customized_title,
                    description=f"Follow-up task after completing: {completed_task.get('title', '')}",
                    task_type=next_type,
                    priority=priority,
                    estimated_hours=hours,
                    relationship=relationship,
                    trigger_reason=f"Logical next step after {task_type}",
                    confidence=0.8,  # High confidence for logical progressions
                    prerequisites=[completed_task.get('id', '')],
                    context_factors=['logical_progression', task_type.lower()],
                    generated_from=completed_task.get('id', 'unknown')
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _pattern_matches_context(self, pattern: ProgressionPattern, 
                                completed_task: Dict[str, Any], 
                                context: ProgressionContext) -> bool:
        """Check if a pattern matches the current context.
        
        Args:
            pattern: Pattern to check
            completed_task: Completed task
            context: Progression context
            
        Returns:
            True if pattern matches
        """
        # Check trigger condition
        if pattern.trigger != ProgressionTrigger.TASK_COMPLETION:
            return False
        
        # Check pattern conditions
        task_type = completed_task.get('type', '').upper()
        repository = completed_task.get('repository', '')
        
        for condition in pattern.conditions:
            if condition.startswith('task_type:'):
                required_type = condition.split(':')[1].upper()
                if task_type != required_type:
                    return False
            elif condition.startswith('repository:'):
                required_repo = condition.split(':')[1]
                if repository != required_repo:
                    return False
            elif condition.startswith('priority:'):
                required_priority = condition.split(':')[1]
                if completed_task.get('priority', '') != required_priority:
                    return False
        
        return True
    
    async def _create_suggestion_from_pattern(self, pattern: ProgressionPattern, 
                                            next_task_type: str,
                                            completed_task: Dict[str, Any], 
                                            context: ProgressionContext) -> Optional[NextTaskSuggestion]:
        """Create a suggestion from a learned pattern.
        
        Args:
            pattern: Source pattern
            next_task_type: Type of next task
            completed_task: Completed task
            context: Progression context
            
        Returns:
            Pattern-based suggestion
        """
        # Generate title and description based on pattern and context
        title_templates = {
            'TESTING': f"Add tests for {completed_task.get('title', 'completed work')}",
            'DOCUMENTATION': f"Update documentation for {completed_task.get('title', 'changes')}",
            'REFACTOR': f"Refactor code related to {completed_task.get('title', 'implementation')}",
            'PERFORMANCE': f"Optimize performance of {completed_task.get('title', 'feature')}",
            'SECURITY': f"Security review of {completed_task.get('title', 'implementation')}"
        }
        
        title = title_templates.get(next_task_type, f"Follow-up task for {completed_task.get('title', '')}")
        
        # Estimate hours based on pattern history and task type
        hour_estimates = {
            'TESTING': 2.0,
            'DOCUMENTATION': 1.0,
            'REFACTOR': 4.0,
            'PERFORMANCE': 3.0,
            'SECURITY': 3.0,
            'BUG_FIX': 2.0,
            'FEATURE': 5.0
        }
        
        estimated_hours = hour_estimates.get(next_task_type, 2.0)
        
        # Determine priority based on pattern and context
        priority = 'medium'
        if pattern.relationship_type == TaskRelationship.SEQUENTIAL:
            priority = 'high'
        elif next_task_type == 'TESTING':
            priority = 'high'
        elif next_task_type in ['SECURITY', 'BUG_FIX']:
            priority = 'high'
        
        suggestion = NextTaskSuggestion(
            title=title,
            description=f"Generated from pattern '{pattern.name}' with {pattern.success_rate:.1%} success rate",
            task_type=next_task_type,
            priority=priority,
            estimated_hours=estimated_hours,
            relationship=pattern.relationship_type,
            trigger_reason=f"Pattern: {pattern.name}",
            confidence=pattern.confidence * pattern.success_rate,
            prerequisites=[completed_task.get('id', '')],
            context_factors=[f"pattern:{pattern.name}"],
            generated_from=completed_task.get('id', 'unknown')
        )
        
        return suggestion
    
    def _deduplicate_suggestions(self, suggestions: List[NextTaskSuggestion]) -> List[NextTaskSuggestion]:
        """Remove duplicate suggestions based on similarity.
        
        Args:
            suggestions: List of suggestions
            
        Returns:
            Deduplicated suggestions
        """
        unique_suggestions = []
        seen_titles = set()
        
        for suggestion in suggestions:
            # Create a normalized version of the title for comparison
            normalized_title = suggestion.title.lower().strip()
            
            # Check for similarity with existing suggestions
            is_duplicate = False
            for seen_title in seen_titles:
                similarity = self._calculate_title_similarity(normalized_title, seen_title)
                if similarity > 0.8:  # 80% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_suggestions.append(suggestion)
                seen_titles.add(normalized_title)
        
        return unique_suggestions
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles.
        
        Args:
            title1: First title
            title2: Second title
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _record_completion(self, completed_task: Dict[str, Any], 
                         context: ProgressionContext) -> None:
        """Record task completion for pattern learning.
        
        Args:
            completed_task: Completed task
            context: Progression context
        """
        completion_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'task': completed_task,
            'context': {
                'repository': context.repository_context.get('name', ''),
                'project_state': context.project_state,
                'priorities': context.current_priorities
            }
        }
        
        self.completion_history.append(completion_record)
        
        # Keep only recent history (last 100 completions)
        if len(self.completion_history) > 100:
            self.completion_history = self.completion_history[-100:]
    
    async def _enhance_context(self, context: ProgressionContext) -> ProgressionContext:
        """Enhance context with aggregated data.
        
        Args:
            context: Basic progression context
            
        Returns:
            Enhanced context
        """
        if self.context_aggregator:
            try:
                aggregated = await self.context_aggregator.gather_comprehensive_context()
                # Add aggregated insights to context
                context.repository_context.update({
                    'cross_repo_patterns': aggregated.cross_repo_patterns,
                    'market_insights': aggregated.market_insights,
                    'external_signals': aggregated.external_signals
                })
                context.current_priorities.extend(aggregated.strategic_priorities)
            except Exception as e:
                self.logger.warning(f"Failed to enhance context: {e}")
        
        return context
    
    async def _generate_from_cross_project_patterns(self, completed_task: Dict[str, Any],
                                                  context: ProgressionContext) -> List[NextTaskSuggestion]:
        """Generate suggestions from cross-project patterns.
        
        Args:
            completed_task: Completed task
            context: Progression context
            
        Returns:
            Cross-project suggestions
        """
        suggestions = []
        task_type = completed_task.get('type', '').upper()
        
        # Find similar completions across projects
        similar_patterns = self._find_cross_project_patterns(task_type)
        
        for pattern in similar_patterns[:2]:  # Top 2 patterns
            if pattern.confidence > 0.7:
                suggestion = NextTaskSuggestion(
                    title=f"Cross-project follow-up: {pattern.next_task_types[0]}",
                    description=f"Based on similar patterns in other projects",
                    task_type=pattern.next_task_types[0],
                    priority='medium',
                    estimated_hours=3.0,
                    relationship=pattern.relationship_type,
                    trigger_reason=f"Cross-project pattern: {pattern.name}",
                    confidence=pattern.confidence * pattern.success_rate,
                    prerequisites=[completed_task.get('id', '')],
                    context_factors=['cross_project_pattern'],
                    generated_from=completed_task.get('id', 'unknown')
                )
                suggestions.append(suggestion)
        
        return suggestions
    
    def _find_cross_project_patterns(self, task_type: str) -> List[ProgressionPattern]:
        """Find patterns from other projects.
        
        Args:
            task_type: Type of completed task
            
        Returns:
            Relevant patterns from other projects
        """
        patterns = []
        
        for pattern_list in self.cross_project_patterns.values():
            for pattern in pattern_list:
                if any(f"task_type:{task_type}" in c for c in pattern.conditions):
                    patterns.append(pattern)
        
        # Sort by success rate and confidence
        patterns.sort(key=lambda p: p.success_rate * p.confidence, reverse=True)
        
        return patterns
    
    async def _enhance_with_predictions(self, suggestions: List[NextTaskSuggestion],
                                      context: ProgressionContext) -> List[NextTaskSuggestion]:
        """Enhance suggestions with predictive insights.
        
        Args:
            suggestions: Base suggestions
            context: Progression context
            
        Returns:
            Enhanced suggestions
        """
        if not self.predictive_engine:
            return suggestions
        
        try:
            # Get predictions
            predictions = await self.predictive_engine.predict_next_tasks(
                context.repository_context
            )
            
            # Map predictions to suggestions
            prediction_map = {p.task_type: p for p in predictions}
            
            # Enhance matching suggestions
            for suggestion in suggestions:
                if suggestion.task_type in prediction_map:
                    prediction = prediction_map[suggestion.task_type]
                    # Boost confidence if predicted
                    suggestion.confidence = min(1.0, suggestion.confidence * 1.2)
                    # Add prediction metadata
                    suggestion.context_factors.append('ml_predicted')
                    suggestion.context_factors.extend(prediction.trigger_factors)
        
        except Exception as e:
            self.logger.warning(f"Failed to enhance with predictions: {e}")
        
        return suggestions
    
    def _adjust_confidence_dynamically(self, suggestions: List[NextTaskSuggestion]) -> List[NextTaskSuggestion]:
        """Adjust confidence based on recent performance.
        
        Args:
            suggestions: Suggestions to adjust
            
        Returns:
            Adjusted suggestions
        """
        for suggestion in suggestions:
            pattern_key = f"{suggestion.task_type}_{suggestion.relationship.value}"
            
            # Get recent performance for this pattern
            if pattern_key in self.pattern_performance_history:
                recent_performance = self.pattern_performance_history[pattern_key][-10:]
                if recent_performance:
                    success_rate = sum(recent_performance) / len(recent_performance)
                    # Adjust confidence based on actual success
                    adjustment = (success_rate - 0.5) * 0.2  # +/- 10% max
                    suggestion.confidence = max(0.1, min(1.0, suggestion.confidence + adjustment))
        
        return suggestions
    
    def _track_pattern_generation(self, completed_task: Dict[str, Any],
                                suggestions: List[NextTaskSuggestion]) -> None:
        """Track pattern generation for learning.
        
        Args:
            completed_task: Completed task
            suggestions: Generated suggestions
        """
        generation_record = {
            'timestamp': datetime.now(timezone.utc),
            'completed_task': completed_task.get('id', 'unknown'),
            'completed_type': completed_task.get('type', ''),
            'suggestions': [{
                'type': s.task_type,
                'relationship': s.relationship.value,
                'confidence': s.confidence
            } for s in suggestions]
        }
        
        # Store for future analysis
        pattern_key = completed_task.get('type', 'unknown')
        self.cross_project_patterns[pattern_key].append(generation_record)
    
    async def _update_pattern_learning(self, completed_task: Dict[str, Any], 
                                     suggestions: List[NextTaskSuggestion],
                                     context: ProgressionContext) -> None:
        """Update pattern learning based on generated suggestions.
        
        Args:
            completed_task: Completed task
            suggestions: Generated suggestions
            context: Progression context
        """
        # Create or update patterns based on suggestions
        for suggestion in suggestions:
            pattern_key = f"{completed_task.get('type', '')}_{suggestion.task_type}_{suggestion.relationship.value}"
            
            if pattern_key not in self.progression_patterns:
                # Create new pattern
                pattern = ProgressionPattern(
                    name=f"Pattern: {completed_task.get('type', '')} → {suggestion.task_type}",
                    trigger=ProgressionTrigger.TASK_COMPLETION,
                    conditions=[f"task_type:{completed_task.get('type', '')}"],
                    next_task_types=[suggestion.task_type],
                    relationship_type=suggestion.relationship,
                    confidence=suggestion.confidence,
                    success_rate=0.5,  # Start with neutral success rate
                    usage_count=1
                )
                self.progression_patterns[pattern_key] = pattern
            else:
                # Update existing pattern
                pattern = self.progression_patterns[pattern_key]
                pattern.usage_count += 1
                # Adjust confidence based on usage
                pattern.confidence = (pattern.confidence + suggestion.confidence) / 2
            
            # Track cross-project patterns if enabled
            if self.cross_project_recognition:
                repo = completed_task.get('repository', 'unknown')
                if repo not in self.cross_project_patterns:
                    self.cross_project_patterns[repo] = []
                self.cross_project_patterns[repo].append(pattern)
    
    def track_suggestion_outcome(self, suggestion_id: str, was_successful: bool,
                               metadata: Dict[str, Any] = None) -> None:
        """Track the outcome of a generated suggestion with enhanced tracking.
        
        Args:
            suggestion_id: ID of the suggestion
            was_successful: Whether the suggestion was successful
            metadata: Additional outcome metadata
        """
        # Track basic outcome
        self.pattern_success_tracking[suggestion_id].append(was_successful)
        
        # Track detailed performance if metadata provided
        if metadata:
            pattern_key = metadata.get('pattern_key', suggestion_id)
            performance_data = {
                'success': was_successful,
                'timestamp': datetime.now(timezone.utc),
                'completion_time': metadata.get('completion_time'),
                'quality_score': metadata.get('quality_score', 0.5)
            }
            self.pattern_performance_history[pattern_key].append(was_successful)
        
        # Update pattern success rates based on tracked outcomes
        self._update_pattern_success_rates()
        
        # Trigger cross-project learning if enabled
        if self.cross_project_recognition and metadata:
            self._update_cross_project_learning(suggestion_id, was_successful, metadata)
    
    def _update_pattern_success_rates(self) -> None:
        """Update pattern success rates based on tracked outcomes."""
        for pattern_key, pattern in self.progression_patterns.items():
            # Get tracked outcomes for this pattern
            if pattern_key in self.pattern_performance_history:
                recent_outcomes = self.pattern_performance_history[pattern_key][-20:]
                if len(recent_outcomes) > 5:
                    # Calculate actual success rate
                    success_count = sum(1 for outcome in recent_outcomes if outcome)
                    new_success_rate = success_count / len(recent_outcomes)
                    
                    # Smooth the update to avoid drastic changes
                    pattern.success_rate = (pattern.success_rate * 0.7 + new_success_rate * 0.3)
                    
                    # Adjust confidence based on consistency
                    consistency = 1.0 - np.std(recent_outcomes)
                    pattern.confidence = min(1.0, pattern.confidence * (0.8 + consistency * 0.2))
            elif pattern.usage_count > 5:
                # Gradual improvement for patterns without detailed tracking
                pattern.success_rate = min(0.9, pattern.success_rate + 0.01)
    
    def _update_cross_project_learning(self, suggestion_id: str, was_successful: bool,
                                     metadata: Dict[str, Any]) -> None:
        """Update cross-project learning from outcomes.
        
        Args:
            suggestion_id: Suggestion ID
            was_successful: Success status
            metadata: Outcome metadata
        """
        task_type = metadata.get('task_type', '')
        source_repo = metadata.get('source_repository', '')
        target_repo = metadata.get('target_repository', '')
        
        if task_type and source_repo != target_repo:
            # This was a cross-project pattern application
            cross_pattern_key = f"{task_type}_cross_project"
            
            # Record the outcome
            cross_outcome = {
                'source': source_repo,
                'target': target_repo,
                'success': was_successful,
                'timestamp': datetime.now(timezone.utc)
            }
            
            if cross_pattern_key not in self.cross_project_patterns:
                self.cross_project_patterns[cross_pattern_key] = []
            
            self.cross_project_patterns[cross_pattern_key].append(cross_outcome)
    
    def get_progression_analytics(self) -> Dict[str, Any]:
        """Get analytics about task progression patterns.
        
        Returns:
            Progression analytics
        """
        analytics = {
            'total_patterns': len(self.progression_patterns),
            'completion_history_size': len(self.completion_history),
            'top_patterns': [],
            'common_progressions': {},
            'success_rates': {},
            'cross_project_insights': {},
            'learning_metrics': {}
        }
        
        # Get top patterns by usage
        sorted_patterns = sorted(
            self.progression_patterns.values(),
            key=lambda p: p.usage_count * p.success_rate,
            reverse=True
        )
        
        analytics['top_patterns'] = [
            {
                'name': p.name,
                'usage_count': p.usage_count,
                'success_rate': p.success_rate,
                'confidence': p.confidence
            }
            for p in sorted_patterns[:5]
        ]
        
        # Analyze common progressions
        progressions = defaultdict(int)
        for completion in self.completion_history[-50:]:  # Last 50 completions
            task_type = completion['task'].get('type', '')
            if task_type:
                progressions[task_type] += 1
        
        analytics['common_progressions'] = dict(progressions)
        
        # Calculate overall success rates
        if self.pattern_success_tracking:
            total_tracked = sum(len(outcomes) for outcomes in self.pattern_success_tracking.values())
            total_successful = sum(sum(outcomes) for outcomes in self.pattern_success_tracking.values())
            analytics['overall_success_rate'] = total_successful / total_tracked if total_tracked > 0 else 0
        
        # Cross-project insights
        if self.cross_project_patterns:
            cross_project_success = []
            for pattern_list in self.cross_project_patterns.values():
                if isinstance(pattern_list, list) and pattern_list:
                    if isinstance(pattern_list[0], dict) and 'success' in pattern_list[0]:
                        successes = [p['success'] for p in pattern_list if 'success' in p]
                        if successes:
                            cross_project_success.extend(successes)
            
            if cross_project_success:
                analytics['cross_project_insights'] = {
                    'total_cross_applications': len(cross_project_success),
                    'success_rate': sum(cross_project_success) / len(cross_project_success),
                    'pattern_count': len(self.cross_project_patterns)
                }
        
        # Learning metrics
        analytics['learning_metrics'] = {
            'dynamic_learning_enabled': self.dynamic_pattern_learning,
            'cross_project_recognition_enabled': self.cross_project_recognition,
            'adaptive_confidence_enabled': self.adaptive_confidence_adjustment,
            'patterns_with_performance_data': len(self.pattern_performance_history)
        }
        
        return analytics
    
    def _load_progression_patterns(self) -> None:
        """Load progression patterns from storage."""
        # This would load patterns from a file or database
        # For now, we'll start with some basic patterns
        
        basic_patterns = [
            ProgressionPattern(
                name="Feature → Testing",
                trigger=ProgressionTrigger.TASK_COMPLETION,
                conditions=["task_type:FEATURE"],
                next_task_types=["TESTING"],
                relationship_type=TaskRelationship.SEQUENTIAL,
                confidence=0.9,
                success_rate=0.8,
                usage_count=0
            ),
            ProgressionPattern(
                name="Bug Fix → Regression Testing",
                trigger=ProgressionTrigger.TASK_COMPLETION,
                conditions=["task_type:BUG_FIX"],
                next_task_types=["TESTING"],
                relationship_type=TaskRelationship.SEQUENTIAL,
                confidence=0.95,
                success_rate=0.9,
                usage_count=0
            ),
            ProgressionPattern(
                name="New Project → Documentation",
                trigger=ProgressionTrigger.TASK_COMPLETION,
                conditions=["task_type:NEW_PROJECT"],
                next_task_types=["DOCUMENTATION"],
                relationship_type=TaskRelationship.MAINTENANCE,
                confidence=0.7,
                success_rate=0.6,
                usage_count=0
            )
        ]
        
        for pattern in basic_patterns:
            key = f"{pattern.conditions[0]}_{pattern.next_task_types[0]}_{pattern.relationship_type.value}"
            self.progression_patterns[key] = pattern
    
    async def suggest_next_actions_for_repository(self, repository_name: str, 
                                                context: Dict[str, Any] = None) -> List[NextTaskSuggestion]:
        """Suggest next actions for a specific repository.
        
        Args:
            repository_name: Target repository
            context: Repository context
            
        Returns:
            Repository-specific task suggestions
        """
        suggestions = []
        
        # Get recently completed tasks for this repository
        recent_completions = [
            completion for completion in self.completion_history[-20:]
            if completion['task'].get('repository') == repository_name
        ]
        
        if not recent_completions:
            # No recent completions, suggest general maintenance tasks
            suggestions.extend(self._generate_maintenance_suggestions(repository_name, context))
        else:
            # Generate based on recent completion patterns
            for completion in recent_completions[-3:]:  # Last 3 completions
                progression_context = ProgressionContext(
                    completed_task=completion['task'],
                    repository_context=context or {},
                    project_state=completion.get('context', {}).get('project_state', {}),
                    recent_patterns=[],
                    current_priorities=[],
                    team_capacity={},
                    timeline_constraints={}
                )
                
                repo_suggestions = await self.generate_next_tasks(
                    completion['task'], 
                    progression_context
                )
                suggestions.extend(repo_suggestions)
        
        # Deduplicate and limit suggestions
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        return unique_suggestions[:5]
    
    def _generate_maintenance_suggestions(self, repository_name: str, 
                                        context: Dict[str, Any] = None) -> List[NextTaskSuggestion]:
        """Generate maintenance suggestions for a repository.
        
        Args:
            repository_name: Repository name
            context: Repository context
            
        Returns:
            Maintenance task suggestions
        """
        suggestions = []
        
        maintenance_tasks = [
            ("Update project dependencies", "DEPENDENCY_UPDATE", 2.0, "high"),
            ("Review and update documentation", "DOCUMENTATION", 3.0, "medium"),
            ("Add missing unit tests", "TESTING", 4.0, "high"),
            ("Security vulnerability scan", "SECURITY", 2.0, "medium"),
            ("Performance analysis and optimization", "PERFORMANCE", 3.0, "low")
        ]
        
        for title, task_type, hours, priority in maintenance_tasks:
            suggestion = NextTaskSuggestion(
                title=f"{title} for {repository_name}",
                description=f"Repository maintenance task: {title.lower()}",
                task_type=task_type,
                priority=priority,
                estimated_hours=hours,
                relationship=TaskRelationship.MAINTENANCE,
                trigger_reason="Repository maintenance schedule",
                confidence=0.6,
                prerequisites=[],
                context_factors=["maintenance", "repository_health"],
                generated_from=f"maintenance_for_{repository_name}"
            )
            suggestions.append(suggestion)
        
        return suggestions