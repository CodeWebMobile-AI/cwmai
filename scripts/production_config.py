"""
Production Configuration for Orchestrated AI System

Manages configuration for all workflow cycles and intervals.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class ExecutionMode(Enum):
    """Execution modes for the orchestrator."""
    DEVELOPMENT = "development"  # Faster cycles for testing
    PRODUCTION = "production"    # Standard intervals
    TEST = "test"               # Single cycle execution
    CUSTOM = "custom"           # User-defined intervals


@dataclass
class CycleConfig:
    """Configuration for a single workflow cycle."""
    name: str
    interval_seconds: int
    enabled: bool = True
    max_duration_seconds: int = 3600  # 1 hour default
    retry_on_failure: bool = True
    max_retries: int = 3
    
    def __post_init__(self):
        """Validate configuration."""
        if self.interval_seconds < 60:
            raise ValueError(f"Interval for {self.name} must be at least 60 seconds")
        if self.max_duration_seconds < self.interval_seconds:
            self.max_duration_seconds = self.interval_seconds


@dataclass
class ProductionConfig:
    """Main configuration for the production orchestrator."""
    mode: ExecutionMode = ExecutionMode.PRODUCTION
    
    # GitHub Configuration
    github_token: str = field(default_factory=lambda: os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT', ''))
    github_repo: str = field(default_factory=lambda: os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai'))
    
    # API Keys
    anthropic_api_key: str = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY', ''))
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    gemini_api_key: str = field(default_factory=lambda: os.getenv('GEMINI_API_KEY', ''))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv('DEEPSEEK_API_KEY', ''))
    
    # Cycle Configurations
    task_cycle: CycleConfig = field(default_factory=lambda: CycleConfig(
        name="task_management",
        interval_seconds=1800,  # 30 minutes
        max_duration_seconds=600  # 10 minutes
    ))
    
    main_cycle: CycleConfig = field(default_factory=lambda: CycleConfig(
        name="main_ai_cycle",
        interval_seconds=14400,  # 4 hours
        max_duration_seconds=3600  # 1 hour
    ))
    
    god_mode_cycle: CycleConfig = field(default_factory=lambda: CycleConfig(
        name="god_mode",
        interval_seconds=21600,  # 6 hours
        max_duration_seconds=7200  # 2 hours
    ))
    
    monitoring_cycle: CycleConfig = field(default_factory=lambda: CycleConfig(
        name="monitoring",
        interval_seconds=86400,  # 24 hours
        max_duration_seconds=1800  # 30 minutes
    ))
    
    research_cycle: CycleConfig = field(default_factory=lambda: CycleConfig(
        name="research_evolution",
        interval_seconds=1800,  # 30 minutes
        max_duration_seconds=900  # 15 minutes
    ))
    
    # General Settings
    max_parallel_operations: int = 3
    enable_auto_commits: bool = False
    enable_issue_creation: bool = True
    log_level: str = "INFO"
    state_backup_interval: int = 3600  # 1 hour
    
    # Safety Settings
    # Self-modification can be enabled via ENABLE_SELF_MODIFICATION env var
    # Set to True to enable the self-improvement system in any mode
    enable_self_modification: bool = False
    safety_threshold: float = 0.8
    require_human_approval: bool = False
    
    # Issue Management Settings
    # Only process issues with 'ai-managed' label when True
    require_ai_managed_label: bool = True
    
    def __post_init__(self):
        """Apply mode-specific configurations."""
        if self.mode == ExecutionMode.DEVELOPMENT:
            self._apply_development_mode()
        elif self.mode == ExecutionMode.TEST:
            self._apply_test_mode()
        elif self.mode == ExecutionMode.CUSTOM:
            # Custom mode uses user-provided values
            pass
            
    def _apply_development_mode(self):
        """Apply development mode settings (faster cycles)."""
        self.task_cycle.interval_seconds = 300  # 5 minutes
        self.main_cycle.interval_seconds = 600  # 10 minutes
        self.god_mode_cycle.interval_seconds = 900  # 15 minutes
        self.monitoring_cycle.interval_seconds = 1800  # 30 minutes
        self.enable_auto_commits = False
        self.enable_issue_creation = False
        
    def _apply_test_mode(self):
        """Apply test mode settings (single execution)."""
        # Set intervals to run once immediately, then wait a long time
        self.task_cycle.interval_seconds = 60    # Run immediately
        self.main_cycle.interval_seconds = 120   # Run after task cycle 
        self.god_mode_cycle.interval_seconds = 180  # Run after main cycle
        self.monitoring_cycle.interval_seconds = 240  # Run after god mode
        self.research_cycle.interval_seconds = 1800  # Keep research active for failures
        self.enable_auto_commits = False
        self.enable_issue_creation = False
        
    def validate(self) -> bool:
        """Validate configuration completeness.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Check required API keys
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
            
        if not self.github_token and self.enable_issue_creation:
            errors.append("GITHUB_TOKEN or CLAUDE_PAT is required for issue creation")
            
        # Validate cycles
        for cycle in [self.task_cycle, self.main_cycle, self.god_mode_cycle, self.monitoring_cycle, self.research_cycle]:
            if cycle.enabled and cycle.interval_seconds < 60:
                errors.append(f"{cycle.name} interval must be at least 60 seconds")
                
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True
        
    def get_enabled_cycles(self) -> Dict[str, CycleConfig]:
        """Get all enabled cycles.
        
        Returns:
            Dictionary of enabled cycles
        """
        cycles = {
            'task': self.task_cycle,
            'main': self.main_cycle,
            'god_mode': self.god_mode_cycle,
            'monitoring': self.monitoring_cycle,
            'research': self.research_cycle
        }
        
        return {name: cycle for name, cycle in cycles.items() if cycle.enabled}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return {
            'mode': self.mode.value,
            'cycles': {
                'task': {
                    'interval': self.task_cycle.interval_seconds,
                    'enabled': self.task_cycle.enabled
                },
                'main': {
                    'interval': self.main_cycle.interval_seconds,
                    'enabled': self.main_cycle.enabled
                },
                'god_mode': {
                    'interval': self.god_mode_cycle.interval_seconds,
                    'enabled': self.god_mode_cycle.enabled
                },
                'monitoring': {
                    'interval': self.monitoring_cycle.interval_seconds,
                    'enabled': self.monitoring_cycle.enabled
                }
            },
            'settings': {
                'max_parallel_operations': self.max_parallel_operations,
                'enable_auto_commits': self.enable_auto_commits,
                'enable_issue_creation': self.enable_issue_creation,
                'enable_self_modification': self.enable_self_modification,
                'safety_threshold': self.safety_threshold
            }
        }


def create_config(mode: Optional[str] = None) -> ProductionConfig:
    """Create configuration based on mode or environment.
    
    Args:
        mode: Optional mode override
        
    Returns:
        Production configuration
    """
    # Determine mode
    if mode:
        execution_mode = ExecutionMode(mode)
    else:
        env_mode = os.getenv('ORCHESTRATOR_MODE', 'production')
        execution_mode = ExecutionMode(env_mode)
        
    # Create config
    config = ProductionConfig(mode=execution_mode)
    
    # Reload API keys from environment (in case they were loaded after import)
    config.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
    config.openai_api_key = os.getenv('OPENAI_API_KEY', '')
    config.gemini_api_key = os.getenv('GEMINI_API_KEY', '')
    config.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', '')
    config.github_token = os.getenv('GITHUB_TOKEN') or os.getenv('CLAUDE_PAT', '')
    config.github_repo = os.getenv('GITHUB_REPOSITORY', 'CodeWebMobile-AI/cwmai')
    
    # Apply any environment overrides
    if os.getenv('TASK_CYCLE_INTERVAL'):
        config.task_cycle.interval_seconds = int(os.getenv('TASK_CYCLE_INTERVAL'))
    if os.getenv('MAIN_CYCLE_INTERVAL'):
        config.main_cycle.interval_seconds = int(os.getenv('MAIN_CYCLE_INTERVAL'))
    if os.getenv('GOD_MODE_INTERVAL'):
        config.god_mode_cycle.interval_seconds = int(os.getenv('GOD_MODE_INTERVAL'))
    if os.getenv('MONITORING_INTERVAL'):
        config.monitoring_cycle.interval_seconds = int(os.getenv('MONITORING_INTERVAL'))
        
    # Safety overrides
    if os.getenv('ENABLE_AUTO_COMMITS'):
        config.enable_auto_commits = os.getenv('ENABLE_AUTO_COMMITS').lower() == 'true'
    if os.getenv('ENABLE_SELF_MODIFICATION'):
        config.enable_self_modification = os.getenv('ENABLE_SELF_MODIFICATION').lower() == 'true'
    
    # Issue management overrides
    if os.getenv('REQUIRE_AI_MANAGED_LABEL'):
        config.require_ai_managed_label = os.getenv('REQUIRE_AI_MANAGED_LABEL').lower() == 'true'
        
    return config