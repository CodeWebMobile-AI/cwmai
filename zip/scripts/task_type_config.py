"""
YAML-based Task Type Configuration with Pydantic Validation

This module provides schema validation and YAML loading for task type definitions.
"""

from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

from scripts.task_types import TaskCategory, TaskComplexity, ArchitectureType


class TaskTypeConfigModel(BaseModel):
    """Pydantic model for task type configuration."""
    
    name: str = Field(..., description="Unique task type identifier")
    category: str = Field(..., description="Task category")
    complexity: str = Field(..., description="Task complexity level")
    typical_duration_hours: float = Field(..., gt=0, description="Typical duration in hours")
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    applicable_stages: List[str] = Field(default_factory=list, description="Applicable lifecycle stages")
    applicable_architectures: List[str] = Field(default_factory=list, description="Applicable architectures")
    prerequisites: List[str] = Field(default_factory=list, description="Task prerequisites")
    success_criteria: Dict[str, Any] = Field(default_factory=dict, description="Success criteria")
    priority_modifier: float = Field(default=1.0, ge=0.1, le=10.0, description="Priority modifier")
    
    # Additional fields for enhanced functionality
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies for DAG")
    estimated_variance: float = Field(default=0.2, ge=0, le=1, description="Variance for PERT analysis")
    automation_potential: float = Field(default=0.0, ge=0, le=1, description="Automation potential score")
    risk_level: str = Field(default="medium", description="Risk level: low, medium, high")
    
    @validator('category')
    def validate_category(cls, v):
        """Validate category against enum."""
        try:
            TaskCategory(v)
        except ValueError:
            valid_categories = [c.value for c in TaskCategory]
            raise ValueError(f"Invalid category. Must be one of: {valid_categories}")
        return v
    
    @validator('complexity')
    def validate_complexity(cls, v):
        """Validate complexity against enum."""
        complexity_map = {
            "trivial": TaskComplexity.TRIVIAL,
            "simple": TaskComplexity.SIMPLE,
            "moderate": TaskComplexity.MODERATE,
            "complex": TaskComplexity.COMPLEX,
            "epic": TaskComplexity.EPIC
        }
        if v.lower() not in complexity_map:
            raise ValueError(f"Invalid complexity. Must be one of: {list(complexity_map.keys())}")
        return v
    
    @validator('applicable_architectures')
    def validate_architectures(cls, v):
        """Validate architectures against enum."""
        valid_architectures = [a.value for a in ArchitectureType]
        for arch in v:
            if arch not in valid_architectures:
                raise ValueError(f"Invalid architecture '{arch}'. Must be one of: {valid_architectures}")
        return v
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        """Validate risk level."""
        valid_levels = ["low", "medium", "high"]
        if v not in valid_levels:
            raise ValueError(f"Invalid risk level. Must be one of: {valid_levels}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "setup_kubernetes_cluster",
                "category": "infrastructure",
                "complexity": "complex",
                "typical_duration_hours": 8.0,
                "required_skills": ["kubernetes", "devops", "networking"],
                "applicable_stages": ["growth", "mature"],
                "applicable_architectures": ["microservices"],
                "prerequisites": ["setup_ci_cd"],
                "dependencies": ["setup_monitoring", "setup_logging"],
                "success_criteria": {
                    "cluster_running": True,
                    "nodes_healthy": True,
                    "ingress_configured": True
                },
                "priority_modifier": 1.3,
                "estimated_variance": 0.3,
                "automation_potential": 0.7,
                "risk_level": "high"
            }
        }


class TaskTypePluginConfigModel(BaseModel):
    """Configuration for a task type plugin."""
    
    plugin_info: Dict[str, Any] = Field(..., description="Plugin metadata")
    task_types: List[TaskTypeConfigModel] = Field(..., description="Task type definitions")
    
    @validator('plugin_info')
    def validate_plugin_info(cls, v):
        """Ensure required plugin info fields."""
        required_fields = ["name", "version", "description"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Plugin info must include '{field}'")
        return v


class YAMLTaskTypeLoader:
    """Loads task type definitions from YAML files."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent / "task_type_configs"
        self.config_dir.mkdir(exist_ok=True)
    
    def load_plugin_config(self, yaml_file: Path) -> TaskTypePluginConfigModel:
        """Load and validate a plugin configuration from YAML."""
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        return TaskTypePluginConfigModel(**data)
    
    def load_all_configs(self) -> Dict[str, TaskTypePluginConfigModel]:
        """Load all YAML configurations from the config directory."""
        configs = {}
        
        for yaml_file in self.config_dir.glob("*.yaml"):
            try:
                config = self.load_plugin_config(yaml_file)
                plugin_name = config.plugin_info["name"]
                configs[plugin_name] = config
            except Exception as e:
                print(f"Error loading config from {yaml_file}: {e}")
        
        return configs
    
    def save_plugin_config(self, config: TaskTypePluginConfigModel, filename: str):
        """Save a plugin configuration to YAML."""
        yaml_file = self.config_dir / filename
        
        with open(yaml_file, 'w') as f:
            yaml.dump(
                config.dict(),
                f,
                default_flow_style=False,
                sort_keys=False
            )


# Example YAML configuration
EXAMPLE_YAML_CONFIG = """
plugin_info:
  name: kubernetes_tasks
  version: 1.0.0
  description: Kubernetes-related task types
  author: DevOps Team

task_types:
  - name: setup_kubernetes_cluster
    category: infrastructure
    complexity: complex
    typical_duration_hours: 8.0
    required_skills:
      - kubernetes
      - devops
      - networking
    applicable_stages:
      - growth
      - mature
    applicable_architectures:
      - microservices
    prerequisites:
      - setup_ci_cd
    dependencies:
      - setup_monitoring
      - setup_logging
    success_criteria:
      cluster_running: true
      nodes_healthy: true
      ingress_configured: true
    priority_modifier: 1.3
    estimated_variance: 0.3
    automation_potential: 0.7
    risk_level: high

  - name: deploy_to_kubernetes
    category: infrastructure
    complexity: moderate
    typical_duration_hours: 2.0
    required_skills:
      - kubernetes
      - docker
    applicable_stages:
      - active_development
      - growth
      - mature
    applicable_architectures:
      - microservices
    prerequisites:
      - setup_kubernetes_cluster
    dependencies: []
    success_criteria:
      deployment_successful: true
      pods_running: true
      service_accessible: true
    priority_modifier: 1.0
    estimated_variance: 0.1
    automation_potential: 0.9
    risk_level: medium
"""