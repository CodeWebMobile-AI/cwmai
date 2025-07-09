"""
Adaptive Parameter Engine

Dynamically adjusts evolutionary parameters based on performance metrics
and system state.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import numpy as np
from datetime import datetime, timedelta
import json


class AdaptiveParameterEngine:
    """Intelligently adapts evolutionary parameters during runtime."""
    
    def __init__(self, config_manager):
        self.logger = logging.getLogger(__name__)
        self.config = config_manager
        
        # Performance tracking
        self.fitness_history = deque(maxlen=50)
        self.diversity_history = deque(maxlen=50)
        self.success_rate_history = deque(maxlen=50)
        self.generation_times = deque(maxlen=20)
        
        # Adaptation state
        self.current_params = {}
        self.adaptation_history = []
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
        
        # Multi-armed bandit for strategy selection
        self.strategy_rewards = {
            "increase_mutation": [],
            "increase_population": [],
            "increase_crossover": [],
            "increase_elite": [],
            "decrease_all": []
        }
        
    def adapt_parameters(self, generation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt parameters based on current generation metrics.
        
        Args:
            generation_metrics: Dict containing:
                - fitness_scores: List of fitness scores
                - diversity_score: Population diversity (0-1)
                - success_rate: Deployment success rate
                - generation_time: Time taken for generation
                - best_fitness: Best fitness in population
                
        Returns:
            Updated parameters dict
        """
        # Update history
        self._update_history(generation_metrics)
        
        # Detect stagnation
        is_stagnating = self._detect_stagnation(generation_metrics.get("best_fitness", 0))
        
        # Compute context for adaptation
        context = {
            "fitness_stagnation": self.stagnation_counter / 10.0,  # Normalized
            "diversity_score": generation_metrics.get("diversity_score", 0.5),
            "fitness_improvement": self._compute_fitness_improvement(),
            "time_pressure": self._compute_time_pressure(generation_metrics.get("generation_time")),
            "success_trend": self._compute_success_trend()
        }
        
        # Select adaptation strategy
        strategy = self._select_adaptation_strategy(context)
        
        # Apply strategy
        new_params = self._apply_strategy(strategy, context)
        
        # Record adaptation
        self.adaptation_history.append({
            "generation": generation_metrics.get("generation", 0),
            "strategy": strategy,
            "context": context,
            "old_params": self.current_params.copy(),
            "new_params": new_params,
            "timestamp": datetime.now().isoformat()
        })
        
        self.current_params = new_params
        return new_params
        
    def _update_history(self, metrics: Dict[str, Any]):
        """Update performance history."""
        if "fitness_scores" in metrics:
            avg_fitness = np.mean(metrics["fitness_scores"])
            self.fitness_history.append(avg_fitness)
            
        if "diversity_score" in metrics:
            self.diversity_history.append(metrics["diversity_score"])
            
        if "success_rate" in metrics:
            self.success_rate_history.append(metrics["success_rate"])
            
        if "generation_time" in metrics:
            self.generation_times.append(metrics["generation_time"])
            
    def _detect_stagnation(self, best_fitness: float) -> bool:
        """Detect if evolution is stagnating."""
        improvement_threshold = 0.01
        
        if best_fitness <= self.last_best_fitness + improvement_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            self.last_best_fitness = best_fitness
            
        return self.stagnation_counter > 3
        
    def _compute_fitness_improvement(self) -> float:
        """Compute recent fitness improvement rate."""
        if len(self.fitness_history) < 2:
            return 0.0
            
        recent = list(self.fitness_history)[-10:]
        if len(recent) < 2:
            return 0.0
            
        # Linear regression slope
        x = np.arange(len(recent))
        slope, _ = np.polyfit(x, recent, 1)
        return float(slope)
        
    def _compute_time_pressure(self, generation_time: Optional[float]) -> float:
        """Compute time pressure based on generation time."""
        if not generation_time or len(self.generation_times) < 2:
            return 0.5
            
        avg_time = np.mean(self.generation_times)
        if generation_time > avg_time * 1.5:
            return 0.8  # High pressure - taking too long
        elif generation_time < avg_time * 0.7:
            return 0.2  # Low pressure - fast generations
        return 0.5
        
    def _compute_success_trend(self) -> float:
        """Compute recent success rate trend."""
        if len(self.success_rate_history) < 3:
            return 0.5
            
        recent = list(self.success_rate_history)[-5:]
        return np.mean(recent)
        
    def _select_adaptation_strategy(self, context: Dict[str, float]) -> str:
        """Select adaptation strategy using multi-armed bandit."""
        # Exploration vs exploitation
        epsilon = 0.2 if self.stagnation_counter < 5 else 0.5  # More exploration when stuck
        
        if np.random.random() < epsilon:
            # Explore: random strategy
            strategy = np.random.choice(list(self.strategy_rewards.keys()))
        else:
            # Exploit: choose best performing strategy
            avg_rewards = {}
            for strat, rewards in self.strategy_rewards.items():
                if rewards:
                    avg_rewards[strat] = np.mean(rewards[-10:])  # Recent performance
                else:
                    avg_rewards[strat] = 0.5  # Neutral prior
                    
            strategy = max(avg_rewards, key=avg_rewards.get)
            
        # Override based on context
        if context["fitness_stagnation"] > 0.7:
            strategy = "increase_mutation"  # Force exploration
        elif context["diversity_score"] < 0.2:
            strategy = "increase_population"  # Need diversity
        elif context["time_pressure"] > 0.7:
            strategy = "decrease_all"  # Speed up
            
        return strategy
        
    def _apply_strategy(self, strategy: str, context: Dict[str, float]) -> Dict[str, Any]:
        """Apply selected adaptation strategy."""
        params = self.current_params.copy()
        
        if strategy == "increase_mutation":
            # Increase mutation rate to explore more
            current = self.config.get_parameter("mutation_rate", context)
            max_val = self.config.config["parameters"]["mutation_rate"]["max"]
            new_val = min(max_val, current * 1.3)
            params["mutation_rate"] = new_val
            
            # Also slightly increase population for diversity
            current_pop = self.config.get_parameter("population_size", context)
            max_pop = self.config.config["parameters"]["population_size"]["max"]
            params["population_size"] = min(max_pop, int(current_pop * 1.1))
            
        elif strategy == "increase_population":
            # Increase population size for diversity
            current = self.config.get_parameter("population_size", context)
            max_val = self.config.config["parameters"]["population_size"]["max"]
            new_val = min(max_val, int(current * 1.5))
            params["population_size"] = new_val
            
            # Adjust elite size proportionally
            elite_ratio = 0.3
            params["elite_size"] = max(1, int(new_val * elite_ratio))
            
        elif strategy == "increase_crossover":
            # Favor crossover over mutation
            current = self.config.get_parameter("crossover_rate", context)
            max_val = self.config.config["parameters"]["crossover_rate"]["max"]
            params["crossover_rate"] = min(max_val, current * 1.2)
            
            # Reduce mutation slightly
            current_mut = self.config.get_parameter("mutation_rate", context)
            params["mutation_rate"] = max(0.1, current_mut * 0.8)
            
        elif strategy == "increase_elite":
            # Preserve more good solutions
            current = self.config.get_parameter("elite_size", context)
            pop_size = params.get("population_size", self.config.get_parameter("population_size", context))
            max_elite = int(pop_size * 0.5)  # Never more than half
            params["elite_size"] = min(max_elite, current + 2)
            
        elif strategy == "decrease_all":
            # Reduce computational load
            for param in ["population_size", "elite_size"]:
                current = self.config.get_parameter(param, context)
                min_val = self.config.config["parameters"][param]["min"]
                if param == "population_size":
                    params[param] = max(min_val, int(current * 0.7))
                else:
                    params[param] = max(min_val, int(current * 0.8))
                    
        # Ensure constraints
        params = self._enforce_constraints(params)
        
        return params
        
    def _enforce_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameters satisfy constraints."""
        # Elite size must be less than population
        if "population_size" in params and "elite_size" in params:
            params["elite_size"] = min(params["elite_size"], int(params["population_size"] * 0.5))
            
        # Tournament size constraints
        if "tournament_size" in params and "population_size" in params:
            params["tournament_size"] = min(params["tournament_size"], params["population_size"] // 2)
            
        # Ensure all parameters are within bounds
        for param, value in params.items():
            if param in self.config.config["parameters"]:
                config = self.config.config["parameters"][param]
                if "min" in config:
                    value = max(value, config["min"])
                if "max" in config:
                    value = min(value, config["max"])
                params[param] = value
                
        return params
        
    def update_strategy_reward(self, strategy: str, reward: float):
        """Update reward for a strategy based on performance."""
        if strategy in self.strategy_rewards:
            self.strategy_rewards[strategy].append(reward)
            # Keep only recent history
            if len(self.strategy_rewards[strategy]) > 50:
                self.strategy_rewards[strategy].pop(0)
                
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptations."""
        return {
            "total_adaptations": len(self.adaptation_history),
            "current_parameters": self.current_params,
            "stagnation_level": self.stagnation_counter,
            "strategy_performance": {
                strat: np.mean(rewards[-10:]) if rewards else 0.5
                for strat, rewards in self.strategy_rewards.items()
            },
            "recent_adaptations": self.adaptation_history[-5:]
        }
        
    def save_adaptation_history(self, filepath: str):
        """Save adaptation history for analysis."""
        data = {
            "history": self.adaptation_history,
            "strategy_rewards": {k: list(v) for k, v in self.strategy_rewards.items()},
            "summary": self.get_adaptation_summary()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)