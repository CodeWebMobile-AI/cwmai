"""
Evolutionary System Configuration

Dynamic configuration system that replaces hard-coded values with adaptive ranges
and learned parameters.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging


class EvolutionaryConfig:
    """Manages dynamic configuration for the evolutionary system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.path.join(
            Path(__file__).parent.parent, 'evolutionary_config.json'
        )
        self.config = self._load_config()
        self.runtime_adaptations = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")
                
        return self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default adaptive configuration."""
        return {
            "version": "2.0",
            "last_updated": datetime.now().isoformat(),
            
            # Adaptive parameter ranges (not fixed values!)
            "parameters": {
                "population_size": {
                    "min": 5,
                    "max": 200,
                    "default": 20,
                    "adaptive": True,
                    "adaptation_rate": 0.1,
                    "description": "Number of improvements per generation"
                },
                "elite_size": {
                    "min": 1,
                    "max": 50,
                    "default": 5,
                    "adaptive": True,
                    "constraint": "must be < population_size * 0.5",
                    "description": "Best improvements preserved"
                },
                "mutation_rate": {
                    "min": 0.05,
                    "max": 0.95,
                    "default": 0.3,
                    "adaptive": True,
                    "adaptation_strategy": "fitness_stagnation",
                    "description": "Probability of mutation"
                },
                "crossover_rate": {
                    "min": 0.1,
                    "max": 0.9,
                    "default": 0.7,
                    "adaptive": True,
                    "description": "Probability of crossover vs mutation"
                },
                "tournament_size": {
                    "min": 2,
                    "max": 10,
                    "default": 3,
                    "adaptive": False,
                    "description": "Selection pressure"
                }
            },
            
            # Dynamic improvement types (discovered and extended at runtime)
            "improvement_types": {
                "core": [
                    "optimization", "error_fix", "feature", "integration"
                ],
                "discovered": [],
                "weights": {},  # Learned preferences
                "embeddings": {}  # For similarity clustering
            },
            
            # Adaptive fitness objectives
            "fitness_objectives": {
                "performance": {
                    "weight": 0.25,
                    "adaptive": True,
                    "metrics": ["response_time", "throughput", "resource_usage"],
                    "learning_rate": 0.05
                },
                "quality": {
                    "weight": 0.3,
                    "adaptive": True,
                    "metrics": ["accuracy", "relevance", "completeness"],
                    "learning_rate": 0.05
                },
                "maintainability": {
                    "weight": 0.2,
                    "adaptive": True,
                    "metrics": ["complexity", "documentation", "test_coverage"],
                    "learning_rate": 0.05
                },
                "innovation": {
                    "weight": 0.25,
                    "adaptive": True,
                    "metrics": ["novelty", "capability_expansion", "user_value"],
                    "learning_rate": 0.05
                },
                "custom": {}  # User-defined objectives
            },
            
            # Strategy profiles
            "profiles": {
                "aggressive": {
                    "population_multiplier": 3.0,
                    "mutation_multiplier": 2.0,
                    "exploration_bias": 0.8
                },
                "conservative": {
                    "population_multiplier": 0.5,
                    "mutation_multiplier": 0.5,
                    "exploration_bias": 0.2
                },
                "balanced": {
                    "population_multiplier": 1.0,
                    "mutation_multiplier": 1.0,
                    "exploration_bias": 0.5
                },
                "adaptive": {
                    "description": "Learns optimal profile over time",
                    "enabled": True
                }
            },
            
            # Model ensemble configuration (no hard-coded preferences!)
            "ensemble": {
                "roles": {
                    "explorer": {
                        "temperature_range": [0.7, 1.0],
                        "preferred_traits": ["creative", "diverse"],
                        "model_selection": "performance_based"
                    },
                    "refiner": {
                        "temperature_range": [0.1, 0.5],
                        "preferred_traits": ["accurate", "detailed"],
                        "model_selection": "performance_based"
                    },
                    "validator": {
                        "temperature_range": [0.0, 0.3],
                        "preferred_traits": ["careful", "thorough"],
                        "model_selection": "performance_based"
                    }
                },
                "model_performance": {},  # Tracked at runtime
                "selection_strategy": "multi_armed_bandit"
            },
            
            # Learning configuration
            "meta_learning": {
                "enabled": True,
                "history_size": 100,
                "pattern_detection": True,
                "transfer_learning": True
            },
            
            # Plugin system
            "plugins": {
                "enabled": True,
                "directories": ["./evolutionary_plugins"],
                "auto_discover": True
            }
        }
        
    def get_parameter(self, name: str, context: Optional[Dict] = None) -> Any:
        """Get parameter value, potentially adapted based on context."""
        if name in self.runtime_adaptations:
            return self.runtime_adaptations[name]
            
        param_config = self.config["parameters"].get(name, {})
        if not param_config:
            return None
            
        # Return default if not adaptive
        if not param_config.get("adaptive", False):
            return param_config.get("default")
            
        # Otherwise, compute adapted value based on context
        return self._compute_adapted_value(name, param_config, context)
        
    def _compute_adapted_value(self, name: str, config: Dict, context: Optional[Dict]) -> Any:
        """Compute adapted parameter value based on context."""
        default = config.get("default")
        if not context:
            return default
            
        # Example adaptations based on context
        if name == "mutation_rate":
            # Increase mutation if fitness is stagnating
            stagnation = context.get("fitness_stagnation", 0)
            adaptation = min(0.3, stagnation * config.get("adaptation_rate", 0.1))
            return min(config["max"], default + adaptation)
            
        elif name == "population_size":
            # Adjust based on available resources and diversity needs
            diversity = context.get("diversity_score", 0.5)
            if diversity < 0.3:  # Low diversity
                return min(config["max"], int(default * 1.5))
            return default
            
        return default
        
    def update_runtime_adaptation(self, name: str, value: Any):
        """Update runtime adaptation for a parameter."""
        self.runtime_adaptations[name] = value
        
    def add_discovered_improvement_type(self, improvement_type: str, metadata: Optional[Dict] = None):
        """Add a newly discovered improvement type."""
        discovered = self.config["improvement_types"]["discovered"]
        if improvement_type not in discovered:
            discovered.append(improvement_type)
            if metadata:
                self.config["improvement_types"]["embeddings"][improvement_type] = metadata
            self.save_config()
            
    def update_objective_weight(self, objective: str, new_weight: float):
        """Update adaptive objective weight based on learning."""
        if objective in self.config["fitness_objectives"]:
            self.config["fitness_objectives"][objective]["weight"] = new_weight
            self.save_config()
            
    def get_profile(self, profile_name: str) -> Dict[str, Any]:
        """Get a strategy profile configuration."""
        return self.config["profiles"].get(profile_name, self.config["profiles"]["balanced"])
        
    def update_model_performance(self, role: str, model: str, performance: float):
        """Update model performance tracking for ensemble."""
        if "model_performance" not in self.config["ensemble"]:
            self.config["ensemble"]["model_performance"] = {}
            
        if role not in self.config["ensemble"]["model_performance"]:
            self.config["ensemble"]["model_performance"][role] = {}
            
        self.config["ensemble"]["model_performance"][role][model] = performance
        self.save_config()
        
    def save_config(self):
        """Save configuration to file."""
        try:
            self.config["last_updated"] = datetime.now().isoformat()
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            
    def get_improvement_types(self, include_discovered: bool = True) -> List[str]:
        """Get all improvement types including discovered ones."""
        types = self.config["improvement_types"]["core"].copy()
        if include_discovered:
            types.extend(self.config["improvement_types"]["discovered"])
        return types
        
    def get_objective_weights(self) -> Dict[str, float]:
        """Get current objective weights."""
        weights = {}
        for obj, config in self.config["fitness_objectives"].items():
            if obj != "custom":
                weights[obj] = config["weight"]
        return weights