"""
Complexity Analyzer

Analyzes task complexity using multiple metrics and AI-driven assessment
to determine optimal decomposition strategies and work breakdown approaches.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
import math

from ai_brain import IntelligentAIBrain


class ComplexityDimension(Enum):
    """Different dimensions of task complexity."""
    TECHNICAL = "technical"         # Technical difficulty and sophistication
    SCOPE = "scope"                # Breadth and scale of work
    DEPENDENCIES = "dependencies"   # Interconnections and prerequisites
    UNCERTAINTY = "uncertainty"     # Unknowns and risk factors
    RESOURCE = "resource"          # Resource requirements and constraints
    TIME = "time"                  # Time-related complexity


class ComplexityLevel(Enum):
    """Overall complexity levels."""
    TRIVIAL = "trivial"           # < 1 hour, single step
    SIMPLE = "simple"             # 1-3 hours, few steps
    MODERATE = "moderate"         # 4-8 hours, multiple steps
    COMPLEX = "complex"           # 8-16 hours, many steps
    VERY_COMPLEX = "very_complex" # 16+ hours, extensive work
    EPIC = "epic"                 # Multi-week effort


@dataclass
class ComplexityScore:
    """Complexity score for a specific dimension."""
    dimension: ComplexityDimension
    score: float  # 0.0 to 1.0
    factors: List[str]
    reasoning: str
    confidence: float  # 0.0 to 1.0


@dataclass
class ComplexityAnalysis:
    """Complete complexity analysis result."""
    task_id: str
    overall_level: ComplexityLevel
    overall_score: float  # 0.0 to 1.0
    dimension_scores: Dict[ComplexityDimension, ComplexityScore]
    decomposition_recommended: bool
    estimated_subtasks: int
    optimal_chunk_size: float  # Hours per subtask
    risk_factors: List[str]
    mitigation_strategies: List[str]
    resource_requirements: Dict[str, Any]
    timeline_estimate: Dict[str, float]
    confidence_level: float


class ComplexityAnalyzer:
    """Analyzes task complexity across multiple dimensions."""
    
    def __init__(self, ai_brain: Optional[IntelligentAIBrain] = None):
        """Initialize the complexity analyzer.
        
        Args:
            ai_brain: Optional AI brain for enhanced analysis
        """
        self.ai_brain = ai_brain
        self.logger = logging.getLogger(__name__)
        
        # Complexity indicators and patterns
        self.technical_indicators = self._load_technical_indicators()
        self.scope_indicators = self._load_scope_indicators()
        self.dependency_patterns = self._load_dependency_patterns()
        self.uncertainty_markers = self._load_uncertainty_markers()
        
        # Complexity thresholds
        self.complexity_thresholds = {
            ComplexityLevel.TRIVIAL: 0.1,
            ComplexityLevel.SIMPLE: 0.3,
            ComplexityLevel.MODERATE: 0.5,
            ComplexityLevel.COMPLEX: 0.7,
            ComplexityLevel.VERY_COMPLEX: 0.9,
            ComplexityLevel.EPIC: 1.0
        }
    
    async def analyze_complexity(self, task: Dict[str, Any], 
                               context: Dict[str, Any] = None) -> ComplexityAnalysis:
        """Analyze the complexity of a task across all dimensions.
        
        Args:
            task: Task to analyze
            context: Additional context (repository, project, etc.)
            
        Returns:
            Complete complexity analysis
        """
        self.logger.info(f"Analyzing complexity for task: {task.get('title', 'Unknown')}")
        
        task_id = task.get('id', 'unknown')
        
        # Analyze each dimension
        dimension_scores = {}
        
        dimension_scores[ComplexityDimension.TECHNICAL] = await self._analyze_technical_complexity(task, context)
        dimension_scores[ComplexityDimension.SCOPE] = await self._analyze_scope_complexity(task, context)
        dimension_scores[ComplexityDimension.DEPENDENCIES] = await self._analyze_dependency_complexity(task, context)
        dimension_scores[ComplexityDimension.UNCERTAINTY] = await self._analyze_uncertainty_complexity(task, context)
        dimension_scores[ComplexityDimension.RESOURCE] = await self._analyze_resource_complexity(task, context)
        dimension_scores[ComplexityDimension.TIME] = await self._analyze_time_complexity(task, context)
        
        # Calculate overall complexity
        overall_score = self._calculate_overall_complexity(dimension_scores)
        overall_level = self._determine_complexity_level(overall_score)
        
        # Determine decomposition recommendation
        decomposition_recommended = overall_score > 0.4 or overall_level in [
            ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX, ComplexityLevel.EPIC
        ]
        
        # Estimate optimal breakdown
        estimated_subtasks, optimal_chunk_size = self._estimate_optimal_breakdown(overall_score, task)
        
        # Identify risk factors and mitigation strategies
        risk_factors = self._identify_risk_factors(dimension_scores, task)
        mitigation_strategies = await self._generate_mitigation_strategies(risk_factors, task, context)
        
        # Estimate resource requirements
        resource_requirements = self._estimate_resource_requirements(dimension_scores, task)
        
        # Create timeline estimate
        timeline_estimate = self._create_timeline_estimate(overall_score, task, estimated_subtasks)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(dimension_scores)
        
        analysis = ComplexityAnalysis(
            task_id=task_id,
            overall_level=overall_level,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            decomposition_recommended=decomposition_recommended,
            estimated_subtasks=estimated_subtasks,
            optimal_chunk_size=optimal_chunk_size,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            resource_requirements=resource_requirements,
            timeline_estimate=timeline_estimate,
            confidence_level=confidence_level
        )
        
        self.logger.info(f"Complexity analysis complete: {overall_level.value} ({overall_score:.2f})")
        return analysis
    
    async def _analyze_technical_complexity(self, task: Dict[str, Any], 
                                          context: Dict[str, Any] = None) -> ComplexityScore:
        """Analyze technical complexity dimension.
        
        Args:
            task: Task to analyze
            context: Additional context
            
        Returns:
            Technical complexity score
        """
        description = task.get('description', '').lower()
        requirements = ' '.join(task.get('requirements', [])).lower()
        task_type = task.get('type', '').upper()
        text = f"{description} {requirements}"
        
        score = 0.0
        factors = []
        
        # Check technical indicators
        for category, indicators in self.technical_indicators.items():
            matches = sum(1 for indicator in indicators if indicator in text)
            if matches > 0:
                category_score = min(1.0, matches * 0.2)
                score += category_score * 0.2  # Each category contributes 20%
                factors.append(f"{category}: {matches} indicators")
        
        # Task type technical complexity
        type_complexity = {
            'NEW_PROJECT': 0.8,
            'FEATURE': 0.6,
            'REFACTOR': 0.7,
            'SECURITY': 0.8,
            'PERFORMANCE': 0.7,
            'BUG_FIX': 0.4,
            'TESTING': 0.5,
            'DOCUMENTATION': 0.3
        }
        
        type_score = type_complexity.get(task_type, 0.5)
        score = (score + type_score) / 2  # Average with type score
        
        # Repository context complexity
        if context and context.get('repository_analysis'):
            repo_analysis = context['repository_analysis']
            tech_stack = repo_analysis.get('technical_stack', {})
            
            # Complex tech stacks increase technical complexity
            tech_count = len(tech_stack.get('languages', [])) + len(tech_stack.get('frameworks', []))
            if tech_count > 5:
                score += 0.1
                factors.append(f"Complex tech stack ({tech_count} technologies)")
        
        # Use AI for enhanced analysis if available
        if self.ai_brain:
            try:
                ai_score = await self._get_ai_technical_assessment(task, context)
                score = (score + ai_score) / 2  # Average with AI assessment
                factors.append("AI-enhanced assessment")
            except Exception as e:
                self.logger.warning(f"AI technical assessment failed: {e}")
        
        reasoning = f"Technical complexity based on: {'; '.join(factors)}"
        
        return ComplexityScore(
            dimension=ComplexityDimension.TECHNICAL,
            score=min(1.0, score),
            factors=factors,
            reasoning=reasoning,
            confidence=0.8 if self.ai_brain else 0.6
        )
    
    async def _analyze_scope_complexity(self, task: Dict[str, Any], 
                                      context: Dict[str, Any] = None) -> ComplexityScore:
        """Analyze scope complexity dimension.
        
        Args:
            task: Task to analyze
            context: Additional context
            
        Returns:
            Scope complexity score
        """
        description = task.get('description', '')
        requirements = task.get('requirements', [])
        estimated_hours = task.get('estimated_hours', 0)
        
        score = 0.0
        factors = []
        
        # Base score from estimated hours
        hour_score = min(1.0, estimated_hours / 20.0)  # 20+ hours = max scope
        score += hour_score * 0.4
        factors.append(f"Estimated hours: {estimated_hours}")
        
        # Requirements count
        req_count = len(requirements)
        req_score = min(1.0, req_count / 10.0)  # 10+ requirements = high scope
        score += req_score * 0.3
        factors.append(f"Requirements count: {req_count}")
        
        # Description length and complexity
        desc_words = len(description.split())
        desc_score = min(1.0, desc_words / 200.0)  # 200+ words = detailed scope
        score += desc_score * 0.2
        factors.append(f"Description complexity: {desc_words} words")
        
        # Scope indicators in text
        text = f"{description} {' '.join(requirements)}".lower()
        scope_matches = sum(1 for indicator in self.scope_indicators if indicator in text)
        scope_score = min(1.0, scope_matches * 0.15)
        score += scope_score * 0.1
        factors.append(f"Scope indicators: {scope_matches}")
        
        reasoning = f"Scope complexity based on: {'; '.join(factors)}"
        
        return ComplexityScore(
            dimension=ComplexityDimension.SCOPE,
            score=min(1.0, score),
            factors=factors,
            reasoning=reasoning,
            confidence=0.9
        )
    
    async def _analyze_dependency_complexity(self, task: Dict[str, Any], 
                                           context: Dict[str, Any] = None) -> ComplexityScore:
        """Analyze dependency complexity dimension.
        
        Args:
            task: Task to analyze
            context: Additional context
            
        Returns:
            Dependency complexity score
        """
        dependencies = task.get('dependencies', [])
        text = f"{task.get('description', '')} {' '.join(task.get('requirements', []))}".lower()
        
        score = 0.0
        factors = []
        
        # Direct dependencies
        dep_count = len(dependencies)
        dep_score = min(1.0, dep_count / 5.0)  # 5+ dependencies = high complexity
        score += dep_score * 0.4
        factors.append(f"Direct dependencies: {dep_count}")
        
        # Dependency pattern indicators
        pattern_matches = 0
        for pattern in self.dependency_patterns:
            if pattern in text:
                pattern_matches += 1
        
        pattern_score = min(1.0, pattern_matches * 0.2)
        score += pattern_score * 0.3
        factors.append(f"Dependency patterns: {pattern_matches}")
        
        # Integration complexity
        integration_keywords = ['integrate', 'interface', 'api', 'connection', 'external', 'third-party']
        integration_matches = sum(1 for keyword in integration_keywords if keyword in text)
        integration_score = min(1.0, integration_matches * 0.25)
        score += integration_score * 0.3
        factors.append(f"Integration indicators: {integration_matches}")
        
        # Repository context dependencies
        if context and context.get('repository_analysis'):
            repo_analysis = context['repository_analysis']
            # Check for external dependencies in repository
            if repo_analysis.get('dependencies_count', 0) > 20:
                score += 0.1
                factors.append("High repository dependency count")
        
        reasoning = f"Dependency complexity based on: {'; '.join(factors)}"
        
        return ComplexityScore(
            dimension=ComplexityDimension.DEPENDENCIES,
            score=min(1.0, score),
            factors=factors,
            reasoning=reasoning,
            confidence=0.7
        )
    
    async def _analyze_uncertainty_complexity(self, task: Dict[str, Any], 
                                            context: Dict[str, Any] = None) -> ComplexityScore:
        """Analyze uncertainty complexity dimension.
        
        Args:
            task: Task to analyze
            context: Additional context
            
        Returns:
            Uncertainty complexity score
        """
        text = f"{task.get('description', '')} {' '.join(task.get('requirements', []))}".lower()
        
        score = 0.0
        factors = []
        
        # Uncertainty markers
        uncertainty_matches = sum(1 for marker in self.uncertainty_markers if marker in text)
        uncertainty_score = min(1.0, uncertainty_matches * 0.3)
        score += uncertainty_score * 0.5
        factors.append(f"Uncertainty markers: {uncertainty_matches}")
        
        # Vague language detection
        vague_words = ['might', 'could', 'should', 'approximately', 'around', 'some', 'various']
        vague_matches = sum(1 for word in vague_words if word in text)
        vague_score = min(1.0, vague_matches * 0.2)
        score += vague_score * 0.3
        factors.append(f"Vague language: {vague_matches}")
        
        # Missing details (short description with high hour estimate)
        estimated_hours = task.get('estimated_hours', 0)
        description_length = len(task.get('description', '').split())
        if estimated_hours > 8 and description_length < 50:
            score += 0.3
            factors.append("High estimate with minimal details")
        
        # Repository context uncertainty
        if context and context.get('repository_analysis'):
            repo_analysis = context['repository_analysis']
            health_score = repo_analysis.get('health_metrics', {}).get('health_score', 100)
            if health_score < 50:
                score += 0.2
                factors.append("Low repository health increases uncertainty")
        
        reasoning = f"Uncertainty complexity based on: {'; '.join(factors)}"
        
        return ComplexityScore(
            dimension=ComplexityDimension.UNCERTAINTY,
            score=min(1.0, score),
            factors=factors,
            reasoning=reasoning,
            confidence=0.6
        )
    
    async def _analyze_resource_complexity(self, task: Dict[str, Any], 
                                         context: Dict[str, Any] = None) -> ComplexityScore:
        """Analyze resource complexity dimension.
        
        Args:
            task: Task to analyze
            context: Additional context
            
        Returns:
            Resource complexity score
        """
        estimated_hours = task.get('estimated_hours', 0)
        text = f"{task.get('description', '')} {' '.join(task.get('requirements', []))}".lower()
        
        score = 0.0
        factors = []
        
        # Time resource complexity
        time_score = min(1.0, estimated_hours / 40.0)  # 40+ hours = high resource need
        score += time_score * 0.4
        factors.append(f"Time requirement: {estimated_hours} hours")
        
        # Skill complexity indicators
        skill_keywords = ['expert', 'specialist', 'advanced', 'senior', 'architect', 'complex']
        skill_matches = sum(1 for keyword in skill_keywords if keyword in text)
        skill_score = min(1.0, skill_matches * 0.3)
        score += skill_score * 0.3
        factors.append(f"Skill requirements: {skill_matches} indicators")
        
        # Tool/environment complexity
        tool_keywords = ['setup', 'configure', 'environment', 'infrastructure', 'deployment']
        tool_matches = sum(1 for keyword in tool_keywords if keyword in text)
        tool_score = min(1.0, tool_matches * 0.25)
        score += tool_score * 0.3
        factors.append(f"Tool/environment needs: {tool_matches}")
        
        reasoning = f"Resource complexity based on: {'; '.join(factors)}"
        
        return ComplexityScore(
            dimension=ComplexityDimension.RESOURCE,
            score=min(1.0, score),
            factors=factors,
            reasoning=reasoning,
            confidence=0.8
        )
    
    async def _analyze_time_complexity(self, task: Dict[str, Any], 
                                     context: Dict[str, Any] = None) -> ComplexityScore:
        """Analyze time complexity dimension.
        
        Args:
            task: Task to analyze
            context: Additional context
            
        Returns:
            Time complexity score
        """
        estimated_hours = task.get('estimated_hours', 0)
        priority = task.get('priority', 'medium')
        text = f"{task.get('description', '')} {' '.join(task.get('requirements', []))}".lower()
        
        score = 0.0
        factors = []
        
        # Base time complexity from estimated hours
        base_score = min(1.0, estimated_hours / 30.0)  # 30+ hours = high time complexity
        score += base_score * 0.5
        factors.append(f"Base estimate: {estimated_hours} hours")
        
        # Sequential vs parallel work indicators
        sequential_keywords = ['step by step', 'in order', 'sequential', 'after', 'then', 'following']
        sequential_matches = sum(1 for keyword in sequential_keywords if keyword in text)
        if sequential_matches > 0:
            score += 0.2
            factors.append(f"Sequential work indicators: {sequential_matches}")
        
        # Time pressure from priority
        priority_multiplier = {'critical': 0.3, 'high': 0.2, 'medium': 0.1, 'low': 0.0}
        priority_score = priority_multiplier.get(priority, 0.1)
        score += priority_score
        factors.append(f"Priority pressure: {priority}")
        
        # Deadline or urgency indicators
        urgency_keywords = ['urgent', 'deadline', 'asap', 'immediately', 'quickly', 'fast']
        urgency_matches = sum(1 for keyword in urgency_keywords if keyword in text)
        if urgency_matches > 0:
            score += 0.2
            factors.append(f"Urgency indicators: {urgency_matches}")
        
        reasoning = f"Time complexity based on: {'; '.join(factors)}"
        
        return ComplexityScore(
            dimension=ComplexityDimension.TIME,
            score=min(1.0, score),
            factors=factors,
            reasoning=reasoning,
            confidence=0.9
        )
    
    def _calculate_overall_complexity(self, dimension_scores: Dict[ComplexityDimension, ComplexityScore]) -> float:
        """Calculate overall complexity score from dimension scores.
        
        Args:
            dimension_scores: Scores for each dimension
            
        Returns:
            Overall complexity score (0.0 to 1.0)
        """
        # Weighted average of dimension scores
        weights = {
            ComplexityDimension.TECHNICAL: 0.25,
            ComplexityDimension.SCOPE: 0.25,
            ComplexityDimension.DEPENDENCIES: 0.20,
            ComplexityDimension.UNCERTAINTY: 0.15,
            ComplexityDimension.RESOURCE: 0.10,
            ComplexityDimension.TIME: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, weight in weights.items():
            if dimension in dimension_scores:
                score = dimension_scores[dimension].score
                confidence = dimension_scores[dimension].confidence
                adjusted_weight = weight * confidence  # Adjust weight by confidence
                weighted_sum += score * adjusted_weight
                total_weight += adjusted_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_complexity_level(self, overall_score: float) -> ComplexityLevel:
        """Determine complexity level from overall score.
        
        Args:
            overall_score: Overall complexity score
            
        Returns:
            Complexity level
        """
        for level in reversed(list(ComplexityLevel)):
            if overall_score >= self.complexity_thresholds[level]:
                return level
        
        return ComplexityLevel.TRIVIAL
    
    def _estimate_optimal_breakdown(self, overall_score: float, task: Dict[str, Any]) -> Tuple[int, float]:
        """Estimate optimal task breakdown.
        
        Args:
            overall_score: Overall complexity score
            task: Task to analyze
            
        Returns:
            Tuple of (estimated_subtasks, optimal_chunk_size_hours)
        """
        estimated_hours = task.get('estimated_hours', 0)
        
        # Base chunk size based on complexity
        if overall_score < 0.3:
            base_chunk_size = 4.0  # Simple tasks: 4 hour chunks
        elif overall_score < 0.6:
            base_chunk_size = 3.0  # Moderate tasks: 3 hour chunks
        elif overall_score < 0.8:
            base_chunk_size = 2.0  # Complex tasks: 2 hour chunks
        else:
            base_chunk_size = 1.5  # Very complex tasks: 1.5 hour chunks
        
        # Calculate estimated subtasks
        estimated_subtasks = max(1, math.ceil(estimated_hours / base_chunk_size))
        
        # Adjust chunk size based on actual breakdown
        optimal_chunk_size = estimated_hours / estimated_subtasks if estimated_subtasks > 0 else base_chunk_size
        
        return estimated_subtasks, optimal_chunk_size
    
    def _identify_risk_factors(self, dimension_scores: Dict[ComplexityDimension, ComplexityScore], 
                             task: Dict[str, Any]) -> List[str]:
        """Identify risk factors based on complexity analysis.
        
        Args:
            dimension_scores: Complexity scores by dimension
            task: Task being analyzed
            
        Returns:
            List of identified risk factors
        """
        risk_factors = []
        
        # High dimension scores indicate risks
        for dimension, score in dimension_scores.items():
            if score.score > 0.7:
                risk_factors.append(f"High {dimension.value} complexity ({score.score:.2f})")
        
        # Specific risk patterns
        if dimension_scores.get(ComplexityDimension.UNCERTAINTY, ComplexityScore(ComplexityDimension.UNCERTAINTY, 0, [], "", 0)).score > 0.6:
            risk_factors.append("High uncertainty may lead to scope creep")
        
        if dimension_scores.get(ComplexityDimension.DEPENDENCIES, ComplexityScore(ComplexityDimension.DEPENDENCIES, 0, [], "", 0)).score > 0.6:
            risk_factors.append("Complex dependencies may cause delays")
        
        if task.get('estimated_hours', 0) > 16:
            risk_factors.append("Large task may be difficult to track progress")
        
        return risk_factors
    
    async def _generate_mitigation_strategies(self, risk_factors: List[str], 
                                            task: Dict[str, Any], 
                                            context: Dict[str, Any] = None) -> List[str]:
        """Generate mitigation strategies for identified risks.
        
        Args:
            risk_factors: Identified risk factors
            task: Task being analyzed
            context: Additional context
            
        Returns:
            List of mitigation strategies
        """
        strategies = []
        
        # Standard mitigation strategies based on risk patterns
        for risk in risk_factors:
            if "uncertainty" in risk.lower():
                strategies.append("Break down into smaller, well-defined sub-tasks")
                strategies.append("Create prototypes or proof-of-concepts first")
            
            elif "dependencies" in risk.lower():
                strategies.append("Identify and prioritize critical dependencies")
                strategies.append("Create parallel work streams where possible")
            
            elif "technical complexity" in risk.lower():
                strategies.append("Allocate extra time for research and learning")
                strategies.append("Consider pair programming or code reviews")
            
            elif "large task" in risk.lower():
                strategies.append("Implement milestone-based progress tracking")
                strategies.append("Regular progress check-ins and adjustments")
        
        # Use AI for additional strategies if available
        if self.ai_brain and risk_factors:
            try:
                ai_strategies = await self._get_ai_mitigation_strategies(risk_factors, task, context)
                strategies.extend(ai_strategies)
            except Exception as e:
                self.logger.warning(f"AI mitigation strategy generation failed: {e}")
        
        return list(set(strategies))  # Remove duplicates
    
    def _estimate_resource_requirements(self, dimension_scores: Dict[ComplexityDimension, ComplexityScore], 
                                      task: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements based on complexity.
        
        Args:
            dimension_scores: Complexity scores by dimension
            task: Task being analyzed
            
        Returns:
            Resource requirements
        """
        requirements = {
            'developer_skill_level': 'intermediate',
            'estimated_team_size': 1,
            'tools_needed': [],
            'infrastructure_requirements': [],
            'external_dependencies': []
        }
        
        # Determine skill level from technical complexity
        tech_score = dimension_scores.get(ComplexityDimension.TECHNICAL, ComplexityScore(ComplexityDimension.TECHNICAL, 0.5, [], "", 0)).score
        
        if tech_score > 0.8:
            requirements['developer_skill_level'] = 'expert'
        elif tech_score > 0.6:
            requirements['developer_skill_level'] = 'senior'
        elif tech_score < 0.3:
            requirements['developer_skill_level'] = 'junior'
        
        # Estimate team size from scope and time complexity
        scope_score = dimension_scores.get(ComplexityDimension.SCOPE, ComplexityScore(ComplexityDimension.SCOPE, 0.5, [], "", 0)).score
        estimated_hours = task.get('estimated_hours', 0)
        
        if scope_score > 0.7 and estimated_hours > 20:
            requirements['estimated_team_size'] = 2
        if scope_score > 0.8 and estimated_hours > 40:
            requirements['estimated_team_size'] = 3
        
        return requirements
    
    def _create_timeline_estimate(self, overall_score: float, task: Dict[str, Any], 
                                estimated_subtasks: int) -> Dict[str, float]:
        """Create timeline estimate based on complexity.
        
        Args:
            overall_score: Overall complexity score
            task: Task being analyzed
            estimated_subtasks: Number of estimated subtasks
            
        Returns:
            Timeline estimate
        """
        base_hours = task.get('estimated_hours', 0)
        
        # Add complexity buffer
        complexity_multiplier = 1.0 + (overall_score * 0.5)  # Up to 50% buffer
        adjusted_hours = base_hours * complexity_multiplier
        
        # Calculate timeline components
        timeline = {
            'base_estimate_hours': base_hours,
            'complexity_adjusted_hours': adjusted_hours,
            'buffer_hours': adjusted_hours - base_hours,
            'estimated_calendar_days': adjusted_hours / 6,  # 6 productive hours per day
            'subtask_average_duration': adjusted_hours / max(1, estimated_subtasks)
        }
        
        return timeline
    
    def _calculate_confidence_level(self, dimension_scores: Dict[ComplexityDimension, ComplexityScore]) -> float:
        """Calculate overall confidence level of the analysis.
        
        Args:
            dimension_scores: Scores for each dimension
            
        Returns:
            Overall confidence level (0.0 to 1.0)
        """
        if not dimension_scores:
            return 0.0
        
        confidence_sum = sum(score.confidence for score in dimension_scores.values())
        return confidence_sum / len(dimension_scores)
    
    async def _get_ai_technical_assessment(self, task: Dict[str, Any], 
                                         context: Dict[str, Any] = None) -> float:
        """Get AI assessment of technical complexity.
        
        Args:
            task: Task to analyze
            context: Additional context
            
        Returns:
            AI technical complexity score (0.0 to 1.0)
        """
        prompt = f"""
        Assess the technical complexity of this task on a scale of 0.0 to 1.0:
        
        Task: {task.get('title', '')}
        Type: {task.get('type', '')}
        Description: {task.get('description', '')}
        Requirements: {json.dumps(task.get('requirements', []))}
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Consider:
        - Technical sophistication required
        - Number of technologies involved
        - Integration complexity
        - Algorithm complexity
        - Architecture decisions needed
        
        Return only a number between 0.0 and 1.0 representing technical complexity.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        content = response.get('content', '0.5')
        
        # Extract number from response
        import re
        number_match = re.search(r'0\.\d+|1\.0|0|1', content)
        if number_match:
            return float(number_match.group())
        
        return 0.5  # Default moderate complexity
    
    async def _get_ai_mitigation_strategies(self, risk_factors: List[str], 
                                          task: Dict[str, Any], 
                                          context: Dict[str, Any] = None) -> List[str]:
        """Get AI-generated mitigation strategies.
        
        Args:
            risk_factors: Identified risk factors
            task: Task being analyzed
            context: Additional context
            
        Returns:
            List of AI-generated mitigation strategies
        """
        prompt = f"""
        Generate mitigation strategies for these task risks:
        
        Task: {task.get('title', '')}
        Risks: {json.dumps(risk_factors)}
        
        Provide 2-3 specific, actionable mitigation strategies.
        Return as a simple list, one strategy per line.
        """
        
        response = await self.ai_brain.generate_enhanced_response(prompt)
        content = response.get('content', '')
        
        # Extract strategies from response
        strategies = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove bullet points and numbering
                cleaned = re.sub(r'^[\d\-\*\.\)]+\s*', '', line)
                if cleaned:
                    strategies.append(cleaned)
        
        return strategies[:3]  # Limit to 3 strategies
    
    def _load_technical_indicators(self) -> Dict[str, List[str]]:
        """Load technical complexity indicators.
        
        Returns:
            Technical indicators by category
        """
        return {
            'algorithms': ['algorithm', 'optimization', 'performance', 'complexity', 'efficient'],
            'architecture': ['architecture', 'design pattern', 'microservices', 'scalable', 'distributed'],
            'integration': ['api', 'integration', 'webhook', 'interface', 'protocol'],
            'security': ['security', 'authentication', 'encryption', 'authorization', 'vulnerability'],
            'data': ['database', 'migration', 'schema', 'query', 'transaction'],
            'infrastructure': ['deployment', 'docker', 'kubernetes', 'cloud', 'infrastructure']
        }
    
    def _load_scope_indicators(self) -> List[str]:
        """Load scope complexity indicators.
        
        Returns:
            List of scope indicators
        """
        return [
            'comprehensive', 'complete', 'full', 'entire', 'all',
            'multiple', 'various', 'several', 'many', 'extensive',
            'end-to-end', 'full-stack', 'across', 'throughout'
        ]
    
    def _load_dependency_patterns(self) -> List[str]:
        """Load dependency complexity patterns.
        
        Returns:
            List of dependency patterns
        """
        return [
            'depends on', 'requires', 'needs', 'must have', 'prerequisite',
            'integrate with', 'connect to', 'interface with', 'work with',
            'external service', 'third party', 'library', 'framework'
        ]
    
    def _load_uncertainty_markers(self) -> List[str]:
        """Load uncertainty complexity markers.
        
        Returns:
            List of uncertainty markers
        """
        return [
            'investigate', 'research', 'explore', 'determine', 'figure out',
            'unclear', 'unknown', 'tbd', 'to be determined', 'needs analysis',
            'prototype', 'poc', 'proof of concept', 'experiment', 'trial'
        ]
    
    def get_complexity_insights(self, analysis: ComplexityAnalysis) -> Dict[str, Any]:
        """Get insights and recommendations from complexity analysis.
        
        Args:
            analysis: Complexity analysis result
            
        Returns:
            Insights and recommendations
        """
        insights = {
            'complexity_summary': {
                'level': analysis.overall_level.value,
                'score': analysis.overall_score,
                'confidence': analysis.confidence_level
            },
            'key_challenges': [],
            'recommendations': [],
            'breakdown_strategy': None,
            'success_factors': []
        }
        
        # Identify key challenges
        for dimension, score in analysis.dimension_scores.items():
            if score.score > 0.7:
                insights['key_challenges'].append(f"High {dimension.value} complexity")
        
        # Generate recommendations
        if analysis.decomposition_recommended:
            insights['recommendations'].append(f"Break into {analysis.estimated_subtasks} sub-tasks")
            insights['breakdown_strategy'] = {
                'estimated_subtasks': analysis.estimated_subtasks,
                'optimal_chunk_size': analysis.optimal_chunk_size,
                'strategy': 'milestone_based' if analysis.estimated_subtasks > 5 else 'sequential'
            }
        
        if analysis.overall_level in [ComplexityLevel.COMPLEX, ComplexityLevel.VERY_COMPLEX]:
            insights['recommendations'].append("Consider pair programming or collaboration")
            insights['recommendations'].append("Implement frequent progress check-ins")
        
        # Success factors
        insights['success_factors'] = [
            "Clear acceptance criteria for each component",
            "Regular progress monitoring and adjustment",
            "Proactive risk mitigation"
        ]
        
        if analysis.confidence_level < 0.7:
            insights['success_factors'].append("Additional requirements gathering and clarification")
        
        return insights