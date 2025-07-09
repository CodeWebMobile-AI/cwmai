"""
Project Context Model for Context-Aware Intelligence

This module defines a comprehensive context model that captures the real-time
state of a project for intelligent task selection and prioritization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum

from scripts.task_types import ArchitectureType


class ProjectStage(Enum):
    """Project lifecycle stages."""
    INCEPTION = "inception"
    EARLY_DEVELOPMENT = "early_development"
    ACTIVE_DEVELOPMENT = "active_development"
    GROWTH = "growth"
    MATURE = "mature"
    MAINTENANCE = "maintenance"
    EOL = "end_of_life"


class TeamMemberRole(Enum):
    """Roles of team members."""
    DEVELOPER = "developer"
    SENIOR_DEVELOPER = "senior_developer"
    LEAD_DEVELOPER = "lead_developer"
    ARCHITECT = "architect"
    DEVOPS = "devops"
    QA = "qa"
    PRODUCT_MANAGER = "product_manager"
    DESIGNER = "designer"


@dataclass
class CodeMetrics:
    """Code quality and complexity metrics."""
    
    # Complexity metrics
    cyclomatic_complexity: float = 0.0
    cognitive_complexity: float = 0.0
    lines_of_code: int = 0
    
    # Quality metrics
    code_coverage: float = 0.0
    technical_debt_ratio: float = 0.0
    maintainability_index: float = 0.0
    
    # Change metrics
    code_churn: float = 0.0  # Rate of code changes
    defect_density: float = 0.0
    
    # Language distribution
    language_distribution: Dict[str, float] = field(default_factory=dict)
    
    # File statistics
    total_files: int = 0
    average_file_size: float = 0.0
    
    # Duplication
    duplication_ratio: float = 0.0


@dataclass
class RepositoryActivity:
    """Repository activity and health metrics."""
    
    # Commit activity
    commits_last_week: int = 0
    commits_last_month: int = 0
    average_commits_per_day: float = 0.0
    
    # Issue tracking
    open_issues: int = 0
    closed_issues_last_week: int = 0
    average_issue_resolution_time: float = 0.0
    critical_bugs: int = 0
    
    # Pull request metrics
    open_pull_requests: int = 0
    average_pr_merge_time: float = 0.0
    pr_rejection_rate: float = 0.0
    
    # Contributor metrics
    active_contributors: int = 0
    new_contributors_last_month: int = 0
    
    # Build/CI metrics
    build_success_rate: float = 0.0
    average_build_time: float = 0.0
    deployment_frequency: float = 0.0


# @dataclass
# class TeamMember:
#     """Team member profile with skills and availability."""
#     
#     id: str
#     name: str
#     role: TeamMemberRole
#     skills: Set[str] = field(default_factory=set)
#     
#     # Availability
#     available_hours_per_week: float = 40.0
#     current_workload: float = 0.0  # Hours committed
#     
#     # Performance metrics
#     task_completion_rate: float = 1.0
#     average_task_duration_ratio: float = 1.0  # Actual vs estimated
#     expertise_areas: Dict[str, float] = field(default_factory=dict)  # Area -> proficiency
#     
#     # Preferences
#     preferred_task_types: List[str] = field(default_factory=list)
#     
#     @property
#     def availability_ratio(self) -> float:
#         """Calculate availability ratio (0-1)."""
#         return max(0, 1 - (self.current_workload / self.available_hours_per_week))


@dataclass
class StrategicGoals:
    """Strategic goals and priorities for the project."""
    
    # Primary goals
    primary_goal: str = ""
    quarterly_objectives: List[str] = field(default_factory=list)
    
    # Focus areas with weights
    focus_areas: Dict[str, float] = field(default_factory=dict)
    
    # Constraints
    deadline: Optional[datetime] = None
    budget_remaining: Optional[float] = None
    
    # Quality targets
    target_code_coverage: float = 0.8
    target_performance_improvement: float = 0.0
    security_priority: float = 1.0  # 0-2 scale
    
    # Business metrics
    user_growth_target: Optional[float] = None
    revenue_target: Optional[float] = None
    customer_satisfaction_target: Optional[float] = None


@dataclass
class TechnicalContext:
    """Technical context and constraints."""
    
    # Architecture
    architecture_type: ArchitectureType
    tech_stack: List[str] = field(default_factory=list)
    
    # Dependencies
    external_dependencies: List[str] = field(default_factory=list)
    dependency_vulnerabilities: int = 0
    outdated_dependencies: int = 0
    
    # Infrastructure
    deployment_environment: str = "development"
    scaling_requirements: str = "low"
    
    # Performance
    current_response_time: float = 0.0
    target_response_time: float = 0.0
    current_throughput: float = 0.0
    
    # Security
    security_score: float = 0.0
    pending_security_patches: int = 0
    compliance_requirements: List[str] = field(default_factory=list)


@dataclass
class ProjectContext:
    """Comprehensive project context for intelligent decision making."""
    
    # Basic info
    project_id: str
    project_name: str
    created_at: datetime
    stage: ProjectStage
    
    # Metrics
    code_metrics: CodeMetrics = field(default_factory=CodeMetrics)
    repository_activity: RepositoryActivity = field(default_factory=RepositoryActivity)
    
    # Team (AI-driven system, no human team members)
    # team_members: List[TeamMember] = field(default_factory=list)
    team_size: int = 0  # AI-only system
    
    # Goals and strategy
    strategic_goals: StrategicGoals = field(default_factory=StrategicGoals)
    
    # Technical context
    technical_context: TechnicalContext = field(default_factory=lambda: TechnicalContext(
        architecture_type=ArchitectureType.LARAVEL_REACT
    ))
    
    # Task history
    completed_task_types: Set[str] = field(default_factory=set)
    failed_task_types: Set[str] = field(default_factory=set)
    task_success_rate: float = 1.0
    
    # Current state
    active_tasks: List[str] = field(default_factory=list)
    blocked_tasks: List[str] = field(default_factory=list)
    
    # External factors
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    competitor_actions: List[str] = field(default_factory=list)
    
    # Computed properties
    @property
    def project_age_days(self) -> int:
        """Calculate project age in days."""
        return (datetime.now() - self.created_at).days
    
    @property
    def project_health_score(self) -> float:
        """Calculate overall project health score (0-1)."""
        scores = []
        
        # Code quality score
        if self.code_metrics.maintainability_index > 0:
            scores.append(min(self.code_metrics.maintainability_index / 100, 1.0))
        
        # Activity score
        if self.repository_activity.commits_last_week > 0:
            activity_score = min(self.repository_activity.commits_last_week / 50, 1.0)
            scores.append(activity_score)
        
        # Team availability score (AI-driven system, always available)
        # if self.team_members:
        #     avg_availability = sum(m.availability_ratio for m in self.team_members) / len(self.team_members)
        #     scores.append(avg_availability)
        scores.append(1.0)  # AI system is always available
        
        # Task success score
        scores.append(self.task_success_rate)
        
        # Build success score
        scores.append(self.repository_activity.build_success_rate)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    @property
    def urgency_level(self) -> float:
        """Calculate urgency level based on various factors (0-1)."""
        urgency_factors = []
        
        # Deadline urgency
        if self.strategic_goals.deadline:
            days_until_deadline = (self.strategic_goals.deadline - datetime.now()).days
            deadline_urgency = max(0, 1 - (days_until_deadline / 90))  # 90 days = 0 urgency
            urgency_factors.append(deadline_urgency)
        
        # Critical bugs urgency
        if self.repository_activity.critical_bugs > 0:
            bug_urgency = min(self.repository_activity.critical_bugs / 10, 1.0)
            urgency_factors.append(bug_urgency)
        
        # Security urgency
        if self.technical_context.pending_security_patches > 0:
            security_urgency = min(self.technical_context.pending_security_patches / 5, 1.0)
            urgency_factors.append(security_urgency)
        
        return max(urgency_factors) if urgency_factors else 0.3
    
    def get_skill_availability(self, required_skill: str) -> float:
        """Get availability of team members with a specific skill."""
        # AI-driven system has all skills available
        return 1.0  # AI can handle any skill requirement
    
    def get_recommended_complexity(self) -> str:
        """Get recommended task complexity based on team state."""
        # AI-driven system can handle any complexity
        avg_workload = 0  # AI system workload is managed differently
        
        if avg_workload > 35:  # High workload
            return "simple"
        elif avg_workload > 25:  # Medium workload
            return "moderate"
        else:  # Low workload
            return "complex"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "stage": self.stage.value,
            "health_score": self.project_health_score,
            "urgency_level": self.urgency_level,
            "team_size": self.team_size,
            "active_tasks": len(self.active_tasks),
            "blocked_tasks": len(self.blocked_tasks),
            "code_metrics": {
                "complexity": self.code_metrics.cyclomatic_complexity,
                "coverage": self.code_metrics.code_coverage,
                "technical_debt": self.code_metrics.technical_debt_ratio
            },
            "repository_activity": {
                "commits_last_week": self.repository_activity.commits_last_week,
                "open_issues": self.repository_activity.open_issues,
                "critical_bugs": self.repository_activity.critical_bugs
            }
        }