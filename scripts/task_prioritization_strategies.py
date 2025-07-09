"""
Dynamic Task Prioritization with Strategy Pattern

This module implements various prioritization strategies that can be
dynamically selected based on project context and goals.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math

from scripts.project_context_model import ProjectContext, ProjectStage
from scripts.task_types import SmartTaskType, TaskComplexity
from scripts.task_type_config import TaskTypeConfigModel


@dataclass
class PrioritizedTask:
    """Task with calculated priority score."""
    task_type: str
    metadata: TaskTypeConfigModel
    priority_score: float
    priority_factors: Dict[str, float]
    
    def __lt__(self, other):
        """Enable sorting by priority score."""
        return self.priority_score > other.priority_score  # Higher score = higher priority


class PrioritizationStrategy(ABC):
    """Base class for all prioritization strategies."""
    
    @abstractmethod
    def calculate_priority(
        self,
        task_type: str,
        metadata: TaskTypeConfigModel,
        context: ProjectContext
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate priority score for a task.
        
        Returns:
            Tuple of (priority_score, factor_breakdown)
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass
    
    def prioritize_tasks(
        self,
        tasks: Dict[str, TaskTypeConfigModel],
        context: ProjectContext
    ) -> List[PrioritizedTask]:
        """Prioritize a list of tasks using this strategy."""
        prioritized = []
        
        for task_type, metadata in tasks.items():
            score, factors = self.calculate_priority(task_type, metadata, context)
            prioritized.append(PrioritizedTask(
                task_type=task_type,
                metadata=metadata,
                priority_score=score,
                priority_factors=factors
            ))
        
        return sorted(prioritized)


class SecurityFirstStrategy(PrioritizationStrategy):
    """Prioritizes security-related tasks highest."""
    
    def get_strategy_name(self) -> str:
        return "security_first"
    
    def calculate_priority(
        self,
        task_type: str,
        metadata: TaskTypeConfigModel,
        context: ProjectContext
    ) -> Tuple[float, Dict[str, float]]:
        factors = {}
        
        # Base priority from metadata
        base_priority = metadata.priority_modifier
        factors["base"] = base_priority
        
        # Security factor
        security_keywords = ["security", "auth", "encryption", "vulnerability", "patch"]
        is_security = any(keyword in task_type.lower() for keyword in security_keywords)
        security_factor = 3.0 if is_security else 1.0
        
        # Boost if there are pending security issues
        if context.technical_context.pending_security_patches > 0:
            security_factor *= 1.5
        
        factors["security"] = security_factor
        
        # Urgency factor based on vulnerabilities
        vulnerability_urgency = min(
            1 + (context.technical_context.dependency_vulnerabilities * 0.2),
            2.0
        )
        factors["vulnerability_urgency"] = vulnerability_urgency
        
        # Risk level factor
        risk_multiplier = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.5
        }.get(metadata.risk_level, 1.0)
        factors["risk"] = risk_multiplier
        
        # Calculate final score
        total_score = base_priority * security_factor * vulnerability_urgency * risk_multiplier
        
        return total_score, factors


class VelocityStrategy(PrioritizationStrategy):
    """Prioritizes tasks that maximize development velocity."""
    
    def get_strategy_name(self) -> str:
        return "velocity"
    
    def calculate_priority(
        self,
        task_type: str,
        metadata: TaskTypeConfigModel,
        context: ProjectContext
    ) -> Tuple[float, Dict[str, float]]:
        factors = {}
        
        # Base priority
        base_priority = metadata.priority_modifier
        factors["base"] = base_priority
        
        # Complexity factor - prefer simpler tasks for velocity
        complexity_factor = {
            TaskComplexity.TRIVIAL: 2.0,
            TaskComplexity.SIMPLE: 1.5,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 0.7,
            TaskComplexity.EPIC: 0.5
        }.get(TaskComplexity[metadata.complexity.upper()], 1.0)
        factors["complexity"] = complexity_factor
        
        # Team availability factor
        required_skills = metadata.required_skills
        skill_availability = sum(
            context.get_skill_availability(skill)
            for skill in required_skills
        ) / len(required_skills) if required_skills else 1.0
        factors["skill_availability"] = 1 + skill_availability
        
        # Automation potential factor
        automation_boost = 1 + (metadata.automation_potential * 0.5)
        factors["automation"] = automation_boost
        
        # Success rate factor
        if task_type in context.failed_task_types:
            factors["past_failure"] = 0.7
        else:
            factors["past_failure"] = 1.0
        
        # Calculate score
        total_score = (
            base_priority * 
            complexity_factor * 
            factors["skill_availability"] * 
            automation_boost * 
            factors["past_failure"]
        )
        
        return total_score, factors


class BusinessValueStrategy(PrioritizationStrategy):
    """Prioritizes tasks based on business value and strategic alignment."""
    
    def get_strategy_name(self) -> str:
        return "business_value"
    
    def calculate_priority(
        self,
        task_type: str,
        metadata: TaskTypeConfigModel,
        context: ProjectContext
    ) -> Tuple[float, Dict[str, float]]:
        factors = {}
        
        # Base priority
        base_priority = metadata.priority_modifier
        factors["base"] = base_priority
        
        # Strategic alignment
        strategic_keywords = {
            "revenue": ["payment", "billing", "subscription", "monetization"],
            "growth": ["feature", "integration", "api", "scaling"],
            "retention": ["performance", "optimization", "ux", "ui"],
            "compliance": ["documentation", "security", "audit", "gdpr"]
        }
        
        alignment_score = 1.0
        for goal, keywords in strategic_keywords.items():
            if goal in context.strategic_goals.focus_areas:
                weight = context.strategic_goals.focus_areas[goal]
                if any(kw in task_type.lower() for kw in keywords):
                    alignment_score += weight
        
        factors["strategic_alignment"] = alignment_score
        
        # Stage appropriateness
        stage_boost = 1.0
        if context.stage.value in metadata.applicable_stages:
            stage_boost = 1.3
        factors["stage_appropriate"] = stage_boost
        
        # Customer impact
        customer_facing = any(
            kw in task_type.lower() 
            for kw in ["ui", "ux", "api", "performance", "feature"]
        )
        customer_impact = 1.5 if customer_facing else 1.0
        factors["customer_impact"] = customer_impact
        
        # ROI estimation (simplified)
        estimated_value = 10 / metadata.typical_duration_hours  # Value per hour
        factors["roi"] = estimated_value
        
        # Calculate score
        total_score = (
            base_priority * 
            alignment_score * 
            stage_boost * 
            customer_impact * 
            estimated_value
        )
        
        return total_score, factors


class TechnicalDebtStrategy(PrioritizationStrategy):
    """Prioritizes tasks that reduce technical debt."""
    
    def get_strategy_name(self) -> str:
        return "technical_debt"
    
    def calculate_priority(
        self,
        task_type: str,
        metadata: TaskTypeConfigModel,
        context: ProjectContext
    ) -> Tuple[float, Dict[str, float]]:
        factors = {}
        
        # Base priority
        base_priority = metadata.priority_modifier
        factors["base"] = base_priority
        
        # Technical debt keywords
        debt_keywords = ["refactor", "cleanup", "optimization", "maintenance", "upgrade"]
        is_debt_reduction = any(kw in task_type.lower() for kw in debt_keywords)
        debt_factor = 2.0 if is_debt_reduction else 1.0
        
        # Boost based on current technical debt
        if context.code_metrics.technical_debt_ratio > 0.2:
            debt_factor *= 1.5
        
        factors["debt_reduction"] = debt_factor
        
        # Code quality impact
        quality_keywords = ["test", "documentation", "lint", "format"]
        quality_impact = 1.5 if any(kw in task_type.lower() for kw in quality_keywords) else 1.0
        factors["quality_impact"] = quality_impact
        
        # Dependency freshness
        if context.technical_context.outdated_dependencies > 5:
            dependency_factor = 1.3
        else:
            dependency_factor = 1.0
        factors["dependency_health"] = dependency_factor
        
        # Calculate score
        total_score = (
            base_priority * 
            debt_factor * 
            quality_impact * 
            dependency_factor
        )
        
        return total_score, factors


class BalancedStrategy(PrioritizationStrategy):
    """Balanced strategy that considers multiple factors equally."""
    
    def get_strategy_name(self) -> str:
        return "balanced"
    
    def calculate_priority(
        self,
        task_type: str,
        metadata: TaskTypeConfigModel,
        context: ProjectContext
    ) -> Tuple[float, Dict[str, float]]:
        # Use all other strategies with equal weight
        strategies = [
            SecurityFirstStrategy(),
            VelocityStrategy(),
            BusinessValueStrategy(),
            TechnicalDebtStrategy()
        ]
        
        all_factors = {}
        scores = []
        
        for strategy in strategies:
            score, factors = strategy.calculate_priority(task_type, metadata, context)
            scores.append(score)
            
            # Prefix factors with strategy name
            for key, value in factors.items():
                all_factors[f"{strategy.get_strategy_name()}_{key}"] = value
        
        # Calculate balanced score (geometric mean for better balance)
        balanced_score = math.pow(math.prod(scores), 1/len(scores)) if scores else 1.0
        all_factors["balanced_score"] = balanced_score
        
        return balanced_score, all_factors


class AdaptiveStrategy(PrioritizationStrategy):
    """Adaptive strategy that changes based on project context."""
    
    def get_strategy_name(self) -> str:
        return "adaptive"
    
    def calculate_priority(
        self,
        task_type: str,
        metadata: TaskTypeConfigModel,
        context: ProjectContext
    ) -> Tuple[float, Dict[str, float]]:
        # Select strategy based on context
        
        # High urgency -> Security first
        if context.urgency_level > 0.7:
            strategy = SecurityFirstStrategy()
        
        # Early stage -> Velocity
        elif context.stage in [ProjectStage.INCEPTION, ProjectStage.EARLY_DEVELOPMENT]:
            strategy = VelocityStrategy()
        
        # Growth stage -> Business value
        elif context.stage in [ProjectStage.GROWTH, ProjectStage.ACTIVE_DEVELOPMENT]:
            strategy = BusinessValueStrategy()
        
        # Mature stage -> Technical debt
        elif context.stage in [ProjectStage.MATURE, ProjectStage.MAINTENANCE]:
            strategy = TechnicalDebtStrategy()
        
        # Default to balanced
        else:
            strategy = BalancedStrategy()
        
        score, factors = strategy.calculate_priority(task_type, metadata, context)
        factors["selected_strategy"] = strategy.get_strategy_name()
        
        return score, factors


# Strategy factory
class PrioritizationStrategyFactory:
    """Factory for creating prioritization strategies."""
    
    _strategies = {
        "security_first": SecurityFirstStrategy,
        "velocity": VelocityStrategy,
        "business_value": BusinessValueStrategy,
        "technical_debt": TechnicalDebtStrategy,
        "balanced": BalancedStrategy,
        "adaptive": AdaptiveStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str) -> PrioritizationStrategy:
        """Create a prioritization strategy by name."""
        strategy_class = cls._strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        return strategy_class()
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names."""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """Register a new strategy type."""
        if not issubclass(strategy_class, PrioritizationStrategy):
            raise TypeError("Strategy must inherit from PrioritizationStrategy")
        cls._strategies[name] = strategy_class