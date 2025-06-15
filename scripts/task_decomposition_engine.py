"""
Task Decomposition Engine

Intelligently breaks down complex tasks into actionable sub-tasks with proper hierarchy,
dependency management, and progressive generation capabilities.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field

from ai_brain import IntelligentAIBrain


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"           # 0-2 hours, single actionable step
    MODERATE = "moderate"       # 2-6 hours, 2-4 steps
    COMPLEX = "complex"         # 6-12 hours, 5-8 steps
    VERY_COMPLEX = "very_complex"  # 12+ hours, 8+ steps, needs decomposition


class DecompositionStrategy(Enum):
    """Strategies for task decomposition."""
    SEQUENTIAL = "sequential"       # Steps must be done in order
    PARALLEL = "parallel"          # Steps can be done concurrently
    HYBRID = "hybrid"              # Mix of sequential and parallel
    MILESTONE_BASED = "milestone_based"  # Based on deliverable milestones


@dataclass
class SubTask:
    """Represents a decomposed sub-task."""
    id: str
    parent_id: str
    title: str
    description: str
    type: str
    priority: str
    estimated_hours: float
    dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    technical_requirements: List[str] = field(default_factory=list)
    sequence_order: int = 0
    can_parallelize: bool = False
    is_blocking: bool = False
    repository_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionResult:
    """Result of task decomposition."""
    original_task_id: str
    sub_tasks: List[SubTask]
    strategy: DecompositionStrategy
    total_estimated_hours: float
    critical_path: List[str]
    parallel_groups: List[List[str]]
    complexity_reduced: bool
    decomposition_rationale: str
    next_actions: List[str]


class TaskDecompositionEngine:
    """Engine for intelligent task decomposition."""
    
    def __init__(self, ai_brain: IntelligentAIBrain):
        """Initialize the decomposition engine.
        
        Args:
            ai_brain: AI brain for intelligent analysis
        """
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Decomposition patterns and templates
        self.decomposition_patterns = self._load_decomposition_patterns()
        self.task_templates = self._load_task_templates()
        
        # Complexity thresholds
        self.complexity_thresholds = {
            TaskComplexity.SIMPLE: 2.0,
            TaskComplexity.MODERATE: 6.0,
            TaskComplexity.COMPLEX: 12.0,
            TaskComplexity.VERY_COMPLEX: float('inf')
        }
        
    def analyze_task_complexity(self, task: Dict[str, Any]) -> TaskComplexity:
        """Analyze task complexity to determine if decomposition is needed.
        
        Args:
            task: Task to analyze
            
        Returns:
            Task complexity level
        """
        estimated_hours = task.get('estimated_hours', 0)
        task_type = task.get('type', '').upper()
        description = task.get('description', '')
        requirements = task.get('requirements', [])
        
        # Base complexity from estimated hours
        if estimated_hours <= 2:
            base_complexity = TaskComplexity.SIMPLE
        elif estimated_hours <= 6:
            base_complexity = TaskComplexity.MODERATE
        elif estimated_hours <= 12:
            base_complexity = TaskComplexity.COMPLEX
        else:
            base_complexity = TaskComplexity.VERY_COMPLEX
        
        # Adjust based on task type
        complexity_multipliers = {
            'NEW_PROJECT': 1.5,
            'FEATURE': 1.2,
            'REFACTOR': 1.3,
            'SECURITY': 1.4,
            'PERFORMANCE': 1.1,
            'BUG_FIX': 0.8,
            'DOCUMENTATION': 0.7,
            'TESTING': 0.9
        }
        
        multiplier = complexity_multipliers.get(task_type, 1.0)
        adjusted_hours = estimated_hours * multiplier
        
        # Check description complexity indicators
        complexity_indicators = [
            'implement', 'create', 'build', 'develop', 'design',
            'integrate', 'configure', 'setup', 'establish',
            'comprehensive', 'complete', 'full-stack', 'end-to-end'
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators 
                            if indicator in description.lower())
        
        # Adjust based on requirements count
        requirement_complexity = len(requirements) * 0.5
        
        # Final complexity calculation
        total_complexity_score = adjusted_hours + indicator_count + requirement_complexity
        
        if total_complexity_score <= 3:
            return TaskComplexity.SIMPLE
        elif total_complexity_score <= 8:
            return TaskComplexity.MODERATE
        elif total_complexity_score <= 15:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX
    
    async def decompose_task(self, task: Dict[str, Any], 
                           repository_context: Dict[str, Any] = None) -> DecompositionResult:
        """Decompose a complex task into actionable sub-tasks.
        
        Args:
            task: Task to decompose
            repository_context: Repository context for targeted decomposition
            
        Returns:
            Decomposition result with sub-tasks
        """
        self.logger.info(f"Decomposing task: {task.get('title', 'Unknown')}")
        
        complexity = self.analyze_task_complexity(task)
        
        if complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
            self.logger.info(f"Task complexity {complexity.value} - no decomposition needed")
            return self._create_minimal_decomposition(task)
        
        # Analyze task for decomposition strategy
        strategy = await self._determine_decomposition_strategy(task, repository_context)
        
        # Generate sub-tasks using AI
        sub_tasks = await self._generate_sub_tasks(task, strategy, repository_context)
        
        # Analyze dependencies and create execution plan
        critical_path, parallel_groups = self._analyze_execution_plan(sub_tasks)
        
        # Calculate total estimated hours
        total_hours = sum(st.estimated_hours for st in sub_tasks)
        
        # Generate rationale and next actions
        rationale = await self._generate_decomposition_rationale(task, sub_tasks, strategy)
        next_actions = self._identify_next_actions(sub_tasks)
        
        result = DecompositionResult(
            original_task_id=task.get('id', ''),
            sub_tasks=sub_tasks,
            strategy=strategy,
            total_estimated_hours=total_hours,
            critical_path=critical_path,
            parallel_groups=parallel_groups,
            complexity_reduced=len(sub_tasks) > 1,
            decomposition_rationale=rationale,
            next_actions=next_actions
        )
        
        self.logger.info(f"Decomposed into {len(sub_tasks)} sub-tasks using {strategy.value} strategy")
        return result
    
    async def _determine_decomposition_strategy(self, task: Dict[str, Any], 
                                              repository_context: Dict[str, Any] = None) -> DecompositionStrategy:
        """Determine the best decomposition strategy for a task.
        
        Args:
            task: Task to analyze
            repository_context: Repository context
            
        Returns:
            Optimal decomposition strategy
        """
        task_type = task.get('type', '').upper()
        description = task.get('description', '')
        
        # Strategy patterns based on task type
        strategy_patterns = {
            'NEW_PROJECT': DecompositionStrategy.MILESTONE_BASED,
            'FEATURE': DecompositionStrategy.SEQUENTIAL,
            'REFACTOR': DecompositionStrategy.HYBRID,
            'SECURITY': DecompositionStrategy.SEQUENTIAL,
            'PERFORMANCE': DecompositionStrategy.HYBRID,
            'BUG_FIX': DecompositionStrategy.SEQUENTIAL,
            'TESTING': DecompositionStrategy.PARALLEL,
            'DOCUMENTATION': DecompositionStrategy.PARALLEL
        }
        
        base_strategy = strategy_patterns.get(task_type, DecompositionStrategy.SEQUENTIAL)
        
        # Use AI to refine strategy based on context
        if self.ai_brain:
            prompt = f"""
            Analyze this task and determine the optimal decomposition strategy:
            
            Task: {task.get('title', '')}
            Type: {task_type}
            Description: {description}
            
            Repository Context: {json.dumps(repository_context or {}, indent=2)}
            
            Available strategies:
            - SEQUENTIAL: Steps must be done in order (setup → implementation → testing)
            - PARALLEL: Steps can be done concurrently (multiple independent features)
            - HYBRID: Mix of sequential and parallel (some dependent, some independent)
            - MILESTONE_BASED: Based on deliverable milestones (MVP → features → polish)
            
            Consider:
            1. Technical dependencies between steps
            2. Resource requirements and team capacity
            3. Risk management and validation points
            4. Repository structure and existing codebase
            
            Return just the strategy name: SEQUENTIAL, PARALLEL, HYBRID, or MILESTONE_BASED
            """
            
            try:
                response = await self.ai_brain.generate_enhanced_response(prompt)
                strategy_name = response.get('content', '').strip().upper()
                
                for strategy in DecompositionStrategy:
                    if strategy.value.upper() == strategy_name:
                        return strategy
                        
            except Exception as e:
                self.logger.warning(f"AI strategy determination failed: {e}")
        
        return base_strategy
    
    async def _generate_sub_tasks(self, task: Dict[str, Any], 
                                strategy: DecompositionStrategy,
                                repository_context: Dict[str, Any] = None) -> List[SubTask]:
        """Generate sub-tasks using AI-driven decomposition.
        
        Args:
            task: Original task
            strategy: Decomposition strategy
            repository_context: Repository context
            
        Returns:
            List of sub-tasks
        """
        prompt = f"""
        Decompose this task into specific, actionable sub-tasks:
        
        Original Task:
        - Title: {task.get('title', '')}
        - Type: {task.get('type', '')}
        - Description: {task.get('description', '')}
        - Estimated Hours: {task.get('estimated_hours', 0)}
        - Requirements: {json.dumps(task.get('requirements', []), indent=2)}
        
        Repository Context: {json.dumps(repository_context or {}, indent=2)}
        Strategy: {strategy.value}
        
        Create sub-tasks that are:
        1. Specific and actionable (1-4 hours each)
        2. Have clear deliverables and acceptance criteria
        3. Include technical requirements
        4. Follow the {strategy.value} strategy
        5. Consider repository structure and tech stack
        
        For each sub-task, provide:
        - title: Clear, actionable title
        - description: Detailed what needs to be done
        - estimated_hours: 0.5-4 hours
        - deliverables: What will be created/modified
        - acceptance_criteria: How to verify completion
        - technical_requirements: Specific technical considerations
        - dependencies: Other sub-tasks this depends on
        - can_parallelize: Whether this can be done in parallel with others
        - sequence_order: Order in the workflow (1, 2, 3...)
        
        Return as JSON array of sub-tasks.
        """
        
        try:
            response = await self.ai_brain.generate_enhanced_response(prompt)
            content = response.get('content', '')
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                sub_tasks_data = json.loads(json_match.group())
                
                sub_tasks = []
                parent_id = task.get('id', 'unknown')
                
                for i, data in enumerate(sub_tasks_data):
                    sub_task = SubTask(
                        id=f"{parent_id}_subtask_{i+1}",
                        parent_id=parent_id,
                        title=data.get('title', f'Sub-task {i+1}'),
                        description=data.get('description', ''),
                        type=data.get('type', task.get('type', '')),
                        priority=self._calculate_subtask_priority(data, task),
                        estimated_hours=float(data.get('estimated_hours', 2.0)),
                        dependencies=data.get('dependencies', []),
                        deliverables=data.get('deliverables', []),
                        acceptance_criteria=data.get('acceptance_criteria', []),
                        technical_requirements=data.get('technical_requirements', []),
                        sequence_order=data.get('sequence_order', i+1),
                        can_parallelize=data.get('can_parallelize', False),
                        repository_context=repository_context or {}
                    )
                    sub_tasks.append(sub_task)
                
                return sub_tasks
                
        except Exception as e:
            self.logger.warning(f"AI sub-task generation failed: {e}")
        
        # Fallback to pattern-based decomposition
        return self._generate_fallback_subtasks(task, strategy)
    
    def _generate_fallback_subtasks(self, task: Dict[str, Any], 
                                  strategy: DecompositionStrategy) -> List[SubTask]:
        """Generate fallback sub-tasks using predefined patterns.
        
        Args:
            task: Original task
            strategy: Decomposition strategy
            
        Returns:
            List of fallback sub-tasks
        """
        task_type = task.get('type', '').upper()
        parent_id = task.get('id', 'unknown')
        
        # Predefined decomposition patterns
        patterns = {
            'NEW_PROJECT': [
                {'title': 'Project Setup and Configuration', 'hours': 2.0, 'order': 1},
                {'title': 'Core Architecture Implementation', 'hours': 4.0, 'order': 2},
                {'title': 'Feature Implementation', 'hours': 6.0, 'order': 3},
                {'title': 'Testing and Validation', 'hours': 2.0, 'order': 4},
                {'title': 'Documentation and Deployment', 'hours': 2.0, 'order': 5}
            ],
            'FEATURE': [
                {'title': 'Design and Planning', 'hours': 1.0, 'order': 1},
                {'title': 'Core Implementation', 'hours': 4.0, 'order': 2},
                {'title': 'Integration and Testing', 'hours': 2.0, 'order': 3},
                {'title': 'Documentation Update', 'hours': 1.0, 'order': 4}
            ],
            'BUG_FIX': [
                {'title': 'Issue Investigation and Root Cause Analysis', 'hours': 1.0, 'order': 1},
                {'title': 'Fix Implementation', 'hours': 2.0, 'order': 2},
                {'title': 'Testing and Validation', 'hours': 1.0, 'order': 3}
            ]
        }
        
        pattern = patterns.get(task_type, patterns['FEATURE'])
        
        sub_tasks = []
        for i, step in enumerate(pattern):
            sub_task = SubTask(
                id=f"{parent_id}_subtask_{i+1}",
                parent_id=parent_id,
                title=step['title'],
                description=f"Complete {step['title'].lower()} for: {task.get('title', '')}",
                type=task.get('type', ''),
                priority=task.get('priority', 'medium'),
                estimated_hours=step['hours'],
                sequence_order=step['order'],
                can_parallelize=step['order'] > 2,  # Later steps can sometimes be parallel
                deliverables=[f"Completed {step['title'].lower()}"],
                acceptance_criteria=[f"{step['title']} meets quality standards"]
            )
            sub_tasks.append(sub_task)
        
        return sub_tasks
    
    def _calculate_subtask_priority(self, subtask_data: Dict[str, Any], 
                                  parent_task: Dict[str, Any]) -> str:
        """Calculate priority for a sub-task.
        
        Args:
            subtask_data: Sub-task data
            parent_task: Parent task
            
        Returns:
            Priority level
        """
        parent_priority = parent_task.get('priority', 'medium')
        
        # Keywords that indicate higher priority
        high_priority_keywords = ['setup', 'core', 'critical', 'foundation', 'security']
        low_priority_keywords = ['documentation', 'cleanup', 'optimization', 'polish']
        
        title = subtask_data.get('title', '').lower()
        
        if any(keyword in title for keyword in high_priority_keywords):
            return 'high' if parent_priority == 'critical' else parent_priority
        elif any(keyword in title for keyword in low_priority_keywords):
            priority_order = ['low', 'medium', 'high', 'critical']
            parent_index = priority_order.index(parent_priority) if parent_priority in priority_order else 1
            return priority_order[max(0, parent_index - 1)]
        else:
            return parent_priority
    
    def _analyze_execution_plan(self, sub_tasks: List[SubTask]) -> Tuple[List[str], List[List[str]]]:
        """Analyze sub-tasks to create execution plan with critical path and parallel groups.
        
        Args:
            sub_tasks: List of sub-tasks
            
        Returns:
            Tuple of (critical_path, parallel_groups)
        """
        # Simple critical path based on sequence order and dependencies
        sequential_tasks = [st for st in sub_tasks if not st.can_parallelize]
        sequential_tasks.sort(key=lambda x: x.sequence_order)
        critical_path = [st.id for st in sequential_tasks]
        
        # Group parallelizable tasks
        parallel_tasks = [st for st in sub_tasks if st.can_parallelize]
        parallel_groups = []
        
        if parallel_tasks:
            # Group by sequence order for parallel execution
            by_order = {}
            for task in parallel_tasks:
                order = task.sequence_order
                if order not in by_order:
                    by_order[order] = []
                by_order[order].append(task.id)
            
            parallel_groups = list(by_order.values())
        
        return critical_path, parallel_groups
    
    async def _generate_decomposition_rationale(self, task: Dict[str, Any], 
                                              sub_tasks: List[SubTask],
                                              strategy: DecompositionStrategy) -> str:
        """Generate rationale for the decomposition approach.
        
        Args:
            task: Original task
            sub_tasks: Generated sub-tasks
            strategy: Used strategy
            
        Returns:
            Decomposition rationale
        """
        rationale = f"""
        Task Decomposition Rationale:
        
        Original Task: {task.get('title', '')} ({task.get('estimated_hours', 0)} hours)
        Strategy Used: {strategy.value}
        Sub-tasks Created: {len(sub_tasks)}
        Total Decomposed Hours: {sum(st.estimated_hours for st in sub_tasks)}
        
        Decomposition Benefits:
        - Improved task granularity for better progress tracking
        - Clear deliverables and acceptance criteria for each step
        - Optimized execution order with parallel opportunities
        - Reduced risk through smaller, manageable chunks
        - Better resource allocation and time estimation
        
        Execution Approach:
        - Sequential tasks follow logical dependencies
        - Parallel tasks can be executed concurrently
        - Each sub-task has specific technical requirements
        - Progress can be tracked at granular level
        """
        
        return rationale.strip()
    
    def _identify_next_actions(self, sub_tasks: List[SubTask]) -> List[str]:
        """Identify immediate next actions from sub-tasks.
        
        Args:
            sub_tasks: List of sub-tasks
            
        Returns:
            List of immediate next actions
        """
        # Find tasks with no dependencies and lowest sequence order
        no_deps = [st for st in sub_tasks if not st.dependencies]
        no_deps.sort(key=lambda x: x.sequence_order)
        
        next_actions = []
        for task in no_deps[:3]:  # Top 3 immediate actions
            next_actions.append(f"Start: {task.title} (Est. {task.estimated_hours}h)")
        
        return next_actions
    
    def _create_minimal_decomposition(self, task: Dict[str, Any]) -> DecompositionResult:
        """Create minimal decomposition for simple tasks.
        
        Args:
            task: Simple task
            
        Returns:
            Minimal decomposition result
        """
        # For simple tasks, create a single sub-task that's essentially the same
        parent_id = task.get('id', 'unknown')
        
        sub_task = SubTask(
            id=f"{parent_id}_subtask_1",
            parent_id=parent_id,
            title=task.get('title', ''),
            description=task.get('description', ''),
            type=task.get('type', ''),
            priority=task.get('priority', 'medium'),
            estimated_hours=task.get('estimated_hours', 2.0),
            deliverables=task.get('requirements', []),
            acceptance_criteria=['Task completed successfully'],
            sequence_order=1
        )
        
        return DecompositionResult(
            original_task_id=parent_id,
            sub_tasks=[sub_task],
            strategy=DecompositionStrategy.SEQUENTIAL,
            total_estimated_hours=task.get('estimated_hours', 2.0),
            critical_path=[sub_task.id],
            parallel_groups=[],
            complexity_reduced=False,
            decomposition_rationale="Task is already appropriately scoped",
            next_actions=[f"Complete: {sub_task.title}"]
        )
    
    def _load_decomposition_patterns(self) -> Dict[str, Any]:
        """Load decomposition patterns from configuration.
        
        Returns:
            Decomposition patterns
        """
        # This could be loaded from a configuration file
        return {
            'complexity_indicators': [
                'implement', 'create', 'build', 'develop', 'design',
                'integrate', 'configure', 'setup', 'establish'
            ],
            'sequential_indicators': [
                'setup', 'configure', 'then', 'after', 'following'
            ],
            'parallel_indicators': [
                'simultaneously', 'concurrently', 'independently', 'separately'
            ]
        }
    
    def _load_task_templates(self) -> Dict[str, Any]:
        """Load task templates for different types.
        
        Returns:
            Task templates
        """
        return {
            'NEW_PROJECT': {
                'phases': ['setup', 'core', 'features', 'testing', 'deployment'],
                'typical_hours': [2, 4, 6, 3, 1]
            },
            'FEATURE': {
                'phases': ['design', 'implement', 'test', 'document'],
                'typical_hours': [1, 4, 2, 1]
            },
            'BUG_FIX': {
                'phases': ['investigate', 'fix', 'test'],
                'typical_hours': [1, 2, 1]
            }
        }
    
    def get_decomposition_analytics(self) -> Dict[str, Any]:
        """Get analytics about decomposition patterns.
        
        Returns:
            Decomposition analytics
        """
        return {
            'complexity_thresholds': {k.value: v for k, v in self.complexity_thresholds.items()},
            'available_strategies': [s.value for s in DecompositionStrategy],
            'supported_task_types': list(self._load_task_templates().keys()),
            'engine_status': 'active'
        }