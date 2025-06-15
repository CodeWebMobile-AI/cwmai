"""
Smart Task Type System with Context Awareness

This module defines an intelligent task type system that considers:
- Project architecture
- Lifecycle stage
- Task complexity and requirements
- Success criteria
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set


class TaskCategory(Enum):
    """Main task categories."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    INFRASTRUCTURE = "infrastructure"
    MAINTENANCE = "maintenance"
    RESEARCH = "research"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"
    SECURITY = "security"


class TaskComplexity(Enum):
    """Task complexity levels."""
    TRIVIAL = 1      # < 1 hour
    SIMPLE = 2       # 1-4 hours
    MODERATE = 3     # 4-8 hours
    COMPLEX = 4      # 1-3 days
    EPIC = 5         # 3+ days


class ArchitectureType(Enum):
    """Common architecture patterns."""
    MONOLITH = "monolith"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    LARAVEL_REACT = "laravel_react"
    DJANGO_VUE = "django_vue"
    NEXTJS = "nextjs"
    API_ONLY = "api_only"
    STATIC_SITE = "static_site"
    CLI_TOOL = "cli_tool"
    LIBRARY = "library"


@dataclass
class TaskTypeMetadata:
    """Metadata for a task type."""
    category: TaskCategory
    complexity: TaskComplexity
    typical_duration_hours: float
    required_skills: Set[str] = field(default_factory=set)
    applicable_stages: Set[str] = field(default_factory=set)
    applicable_architectures: Set[ArchitectureType] = field(default_factory=set)
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    priority_modifier: float = 1.0  # Multiplier for priority calculation
    

class SmartTaskType(Enum):
    """Intelligent task types with contextual awareness."""
    
    # Setup Tasks (Inception Stage)
    SETUP_PROJECT_STRUCTURE = "setup_project_structure"
    SETUP_DATABASE_SCHEMA = "setup_database_schema"
    SETUP_AUTHENTICATION = "setup_authentication"
    SETUP_CI_CD = "setup_ci_cd"
    SETUP_DEVELOPMENT_ENV = "setup_development_env"
    
    # Laravel/React Specific Setup
    SETUP_LARAVEL_API = "setup_laravel_api"
    SETUP_REACT_COMPONENTS = "setup_react_components"
    SETUP_SANCTUM_AUTH = "setup_sanctum_auth"
    
    # Core Development (Early Development Stage)
    FEATURE_CRUD_OPERATIONS = "feature_crud_operations"
    FEATURE_API_ENDPOINT = "feature_api_endpoint"
    FEATURE_UI_COMPONENT = "feature_ui_component"
    FEATURE_BUSINESS_LOGIC = "feature_business_logic"
    FEATURE_DATA_MODEL = "feature_data_model"
    
    # Testing Tasks
    TESTING_UNIT_TESTS = "testing_unit_tests"
    TESTING_INTEGRATION_TESTS = "testing_integration_tests"
    TESTING_E2E_TESTS = "testing_e2e_tests"
    TESTING_API_TESTS = "testing_api_tests"
    TESTING_COMPONENT_TESTS = "testing_component_tests"
    
    # Documentation Tasks
    DOCUMENTATION_API = "documentation_api"
    DOCUMENTATION_README = "documentation_readme"
    DOCUMENTATION_ARCHITECTURE = "documentation_architecture"
    DOCUMENTATION_USER_GUIDE = "documentation_user_guide"
    DOCUMENTATION_DEVELOPER_GUIDE = "documentation_developer_guide"
    
    # Infrastructure Tasks
    INFRASTRUCTURE_DEPLOYMENT = "infrastructure_deployment"
    INFRASTRUCTURE_MONITORING = "infrastructure_monitoring"
    INFRASTRUCTURE_SCALING = "infrastructure_scaling"
    INFRASTRUCTURE_BACKUP = "infrastructure_backup"
    
    # Optimization Tasks (Growth/Mature Stage)
    OPTIMIZATION_PERFORMANCE = "optimization_performance"
    OPTIMIZATION_DATABASE = "optimization_database"
    OPTIMIZATION_CACHING = "optimization_caching"
    OPTIMIZATION_BUNDLE_SIZE = "optimization_bundle_size"
    
    # Maintenance Tasks
    MAINTENANCE_DEPENDENCIES = "maintenance_dependencies"
    MAINTENANCE_SECURITY_PATCH = "maintenance_security_patch"
    MAINTENANCE_BUG_FIX = "maintenance_bug_fix"
    MAINTENANCE_REFACTORING = "maintenance_refactoring"
    
    # Integration Tasks
    INTEGRATION_API = "integration_api"
    INTEGRATION_THIRD_PARTY = "integration_third_party"
    INTEGRATION_PAYMENT = "integration_payment"
    INTEGRATION_ANALYTICS = "integration_analytics"
    
    # Research Tasks
    RESEARCH_TECHNOLOGY = "research_technology"
    RESEARCH_MARKET = "research_market"
    RESEARCH_COMPETITOR = "research_competitor"
    RESEARCH_USER_NEEDS = "research_user_needs"
    
    # Special Tasks
    NEW_PROJECT = "new_project"
    REPOSITORY_HEALTH = "repository_health"
    STRATEGIC_INITIATIVE = "strategic_initiative"


# Task Type Registry with metadata
TASK_TYPE_REGISTRY: Dict[SmartTaskType, TaskTypeMetadata] = {
    # Setup Tasks
    SmartTaskType.SETUP_PROJECT_STRUCTURE: TaskTypeMetadata(
        category=TaskCategory.DEVELOPMENT,
        complexity=TaskComplexity.SIMPLE,
        typical_duration_hours=2,
        required_skills={"project_setup", "architecture"},
        applicable_stages={"inception"},
        applicable_architectures={ArchitectureType.MONOLITH, ArchitectureType.LARAVEL_REACT},
        success_criteria={"has_structure": True, "can_run": True}
    ),
    
    SmartTaskType.SETUP_LARAVEL_API: TaskTypeMetadata(
        category=TaskCategory.DEVELOPMENT,
        complexity=TaskComplexity.MODERATE,
        typical_duration_hours=4,
        required_skills={"laravel", "php", "api_design"},
        applicable_stages={"inception", "early_development"},
        applicable_architectures={ArchitectureType.LARAVEL_REACT},
        prerequisites=[],  # No prerequisites for basic setup
        success_criteria={"api_responds": True, "routes_defined": True}
    ),
    
    SmartTaskType.SETUP_SANCTUM_AUTH: TaskTypeMetadata(
        category=TaskCategory.DEVELOPMENT,
        complexity=TaskComplexity.MODERATE,
        typical_duration_hours=3,
        required_skills={"laravel", "authentication", "security"},
        applicable_stages={"inception", "early_development"},
        applicable_architectures={ArchitectureType.LARAVEL_REACT},
        prerequisites=["setup_laravel_api"],
        success_criteria={"auth_works": True, "tokens_issued": True}
    ),
    
    # Feature Development
    SmartTaskType.FEATURE_API_ENDPOINT: TaskTypeMetadata(
        category=TaskCategory.DEVELOPMENT,
        complexity=TaskComplexity.MODERATE,
        typical_duration_hours=4,
        required_skills={"api_design", "backend"},
        applicable_stages={"early_development", "active_development", "growth"},
        applicable_architectures={ArchitectureType.LARAVEL_REACT, ArchitectureType.API_ONLY},
        success_criteria={"endpoint_works": True, "documented": True}
    ),
    
    SmartTaskType.FEATURE_UI_COMPONENT: TaskTypeMetadata(
        category=TaskCategory.DEVELOPMENT,
        complexity=TaskComplexity.MODERATE,
        typical_duration_hours=4,
        required_skills={"react", "typescript", "ui_design"},
        applicable_stages={"early_development", "active_development"},
        applicable_architectures={ArchitectureType.LARAVEL_REACT, ArchitectureType.NEXTJS},
        success_criteria={"component_renders": True, "responsive": True}
    ),
    
    # Testing Tasks
    SmartTaskType.TESTING_UNIT_TESTS: TaskTypeMetadata(
        category=TaskCategory.TESTING,
        complexity=TaskComplexity.SIMPLE,
        typical_duration_hours=2,
        required_skills={"testing", "unit_testing"},
        applicable_stages={"early_development", "active_development", "growth", "mature"},
        success_criteria={"coverage_increased": True, "tests_pass": True},
        priority_modifier=1.2  # Higher priority for testing
    ),
    
    # Documentation Tasks
    SmartTaskType.DOCUMENTATION_API: TaskTypeMetadata(
        category=TaskCategory.DOCUMENTATION,
        complexity=TaskComplexity.SIMPLE,
        typical_duration_hours=2,
        required_skills={"technical_writing", "api_documentation"},
        applicable_stages={"active_development", "growth", "mature"},
        applicable_architectures={ArchitectureType.LARAVEL_REACT, ArchitectureType.API_ONLY},
        success_criteria={"api_documented": True, "examples_provided": True}
    ),
    
    # Optimization Tasks
    SmartTaskType.OPTIMIZATION_PERFORMANCE: TaskTypeMetadata(
        category=TaskCategory.OPTIMIZATION,
        complexity=TaskComplexity.COMPLEX,
        typical_duration_hours=8,
        required_skills={"performance_optimization", "profiling"},
        applicable_stages={"growth", "mature"},
        success_criteria={"performance_improved": True, "metrics_tracked": True}
    ),
    
    # Maintenance Tasks
    SmartTaskType.MAINTENANCE_SECURITY_PATCH: TaskTypeMetadata(
        category=TaskCategory.MAINTENANCE,
        complexity=TaskComplexity.MODERATE,
        typical_duration_hours=3,
        required_skills={"security", "patching"},
        applicable_stages={"growth", "mature", "maintenance"},
        success_criteria={"vulnerabilities_fixed": True, "tests_pass": True},
        priority_modifier=1.5  # Higher priority for security
    ),
    
    SmartTaskType.NEW_PROJECT: TaskTypeMetadata(
        category=TaskCategory.DEVELOPMENT,
        complexity=TaskComplexity.COMPLEX,
        typical_duration_hours=8,
        required_skills={"project_setup", "architecture", "planning"},
        applicable_stages={"inception"},
        success_criteria={"project_created": True, "can_run": True}
    ),
    
    SmartTaskType.REPOSITORY_HEALTH: TaskTypeMetadata(
        category=TaskCategory.MAINTENANCE,
        complexity=TaskComplexity.MODERATE,
        typical_duration_hours=3,
        required_skills={"analysis", "debugging"},
        applicable_stages={"early_development", "active_development", "growth", "mature"},
        success_criteria={"health_improved": True}
    )
}


class TaskTypeSelector:
    """Intelligent task type selection based on context."""
    
    @staticmethod
    def get_appropriate_task_types(
        architecture: Optional[ArchitectureType],
        lifecycle_stage: str,
        current_needs: List[str],
        completed_tasks: Set[str]
    ) -> List[SmartTaskType]:
        """Get task types appropriate for the current context.
        
        Args:
            architecture: Project architecture type
            lifecycle_stage: Current lifecycle stage
            current_needs: Identified needs from analysis
            completed_tasks: Set of already completed task types
            
        Returns:
            List of appropriate task types
        """
        appropriate_types = []
        
        for task_type, metadata in TASK_TYPE_REGISTRY.items():
            # Check if task is applicable to current stage
            if lifecycle_stage not in metadata.applicable_stages:
                continue
                
            # Check if task is applicable to architecture
            if architecture and metadata.applicable_architectures:
                if architecture not in metadata.applicable_architectures:
                    continue
            
            # Check prerequisites
            if metadata.prerequisites:
                if not all(prereq in completed_tasks for prereq in metadata.prerequisites):
                    continue
            
            # Check if already completed
            if task_type.value in completed_tasks:
                continue
                
            appropriate_types.append(task_type)
        
        # Sort by priority and complexity
        appropriate_types.sort(
            key=lambda t: (
                -TASK_TYPE_REGISTRY[t].priority_modifier,
                TASK_TYPE_REGISTRY[t].complexity.value
            )
        )
        
        return appropriate_types
    
    @staticmethod
    def get_task_metadata(task_type: SmartTaskType) -> TaskTypeMetadata:
        """Get metadata for a task type."""
        return TASK_TYPE_REGISTRY.get(task_type, TaskTypeMetadata(
            category=TaskCategory.DEVELOPMENT,
            complexity=TaskComplexity.MODERATE,
            typical_duration_hours=4
        ))
    
    @staticmethod
    def estimate_task_duration(
        task_type: SmartTaskType,
        repository_complexity: float = 1.0
    ) -> int:
        """Estimate task duration in cycles.
        
        Args:
            task_type: The task type
            repository_complexity: Complexity multiplier for the repository
            
        Returns:
            Estimated cycles (where 1 cycle ≈ 1 hour)
        """
        metadata = TaskTypeSelector.get_task_metadata(task_type)
        base_hours = metadata.typical_duration_hours
        
        # Adjust for repository complexity
        adjusted_hours = base_hours * repository_complexity
        
        # Convert to cycles (rounded up)
        return max(1, int(adjusted_hours + 0.5))


def get_task_type_for_string(task_string: str) -> Optional[SmartTaskType]:
    """Convert a string task type to SmartTaskType.
    
    Args:
        task_string: String representation of task type
        
    Returns:
        SmartTaskType or None if not found
    """
    # Try direct match
    try:
        return SmartTaskType(task_string.lower())
    except ValueError:
        pass
    
    # Try mapping common strings
    string_mappings = {
        "feature": SmartTaskType.FEATURE_BUSINESS_LOGIC,
        "testing": SmartTaskType.TESTING_UNIT_TESTS,
        "documentation": SmartTaskType.DOCUMENTATION_README,
        "optimization": SmartTaskType.OPTIMIZATION_PERFORMANCE,
        "maintenance": SmartTaskType.MAINTENANCE_BUG_FIX,
        "infrastructure": SmartTaskType.INFRASTRUCTURE_DEPLOYMENT,
        "integration": SmartTaskType.INTEGRATION_API,
        "research": SmartTaskType.RESEARCH_TECHNOLOGY,
        "repository_health": SmartTaskType.REPOSITORY_HEALTH,
        "new_project": SmartTaskType.NEW_PROJECT,
    }
    
    task_lower = task_string.lower()
    for key, value in string_mappings.items():
        if key in task_lower:
            return value
    
    return None