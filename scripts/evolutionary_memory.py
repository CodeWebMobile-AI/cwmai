"""
Evolutionary Memory System

Implements meta-learning by tracking patterns across runs and learning
from historical performance data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import logging
import pickle


class EvolutionaryMemory:
    """Long-term memory system for evolutionary learning."""
    
    def __init__(self, memory_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.memory_dir = Path(memory_dir or "evolutionary_memory")
        self.memory_dir.mkdir(exist_ok=True)
        
        # Memory components
        self.patterns_memory = self._load_memory("patterns")
        self.performance_memory = self._load_memory("performance")
        self.strategies_memory = self._load_memory("strategies")
        self.failures_memory = self._load_memory("failures")
        
        # Active session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_data = {
            "start_time": datetime.now().isoformat(),
            "improvements": [],
            "parameters_used": [],
            "strategies_used": [],
            "outcomes": []
        }
        
    def _load_memory(self, memory_type: str) -> Dict:
        """Load a specific memory component from disk."""
        memory_file = self.memory_dir / f"{memory_type}_memory.json"
        if memory_file.exists():
            try:
                with open(memory_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load {memory_type} memory: {e}")
        
        # Initialize with defaults
        if memory_type == "patterns":
            return {
                "successful_patterns": defaultdict(list),
                "improvement_sequences": [],
                "parameter_patterns": defaultdict(list)
            }
        elif memory_type == "performance":
            return {
                "best_configurations": [],
                "performance_by_context": defaultdict(list),
                "optimization_history": []
            }
        elif memory_type == "strategies":
            return {
                "strategy_success_rates": defaultdict(list),
                "context_strategy_map": defaultdict(list),
                "ensemble_performances": []
            }
        elif memory_type == "failures":
            return {
                "common_failures": defaultdict(int),
                "failure_patterns": [],
                "recovery_strategies": defaultdict(list)
            }
        
        return {}
        
    def remember_improvement(self, improvement: Dict[str, Any], outcome: Dict[str, Any]):
        """Remember an improvement and its outcome."""
        memory_entry = {
            "improvement": improvement,
            "outcome": outcome,
            "context": self._get_current_context(),
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        
        # Update session data
        self.session_data["improvements"].append(memory_entry)
        
        # Extract patterns
        if outcome.get("success", False):
            # Successful pattern
            pattern_key = f"{improvement['type']}_{improvement.get('approach', 'default')}"
            self.patterns_memory["successful_patterns"][pattern_key].append({
                "fitness": outcome.get("fitness", 0),
                "metrics": outcome.get("metrics", {}),
                "context": memory_entry["context"]
            })
            
        else:
            # Failure pattern
            failure_key = f"{improvement['type']}_{outcome.get('error_type', 'unknown')}"
            self.failures_memory["common_failures"][failure_key] += 1
            
    def remember_parameters(self, parameters: Dict[str, Any], performance: float):
        """Remember parameter configuration and its performance."""
        param_entry = {
            "parameters": parameters,
            "performance": performance,
            "context": self._get_current_context(),
            "timestamp": datetime.now().isoformat()
        }
        
        self.session_data["parameters_used"].append(param_entry)
        
        # Track best configurations
        self.performance_memory["best_configurations"].append(param_entry)
        self.performance_memory["best_configurations"].sort(
            key=lambda x: x["performance"], 
            reverse=True
        )
        # Keep only top 100
        self.performance_memory["best_configurations"] = \
            self.performance_memory["best_configurations"][:100]
            
    def remember_strategy(self, strategy: str, context: Dict, reward: float):
        """Remember strategy performance in context."""
        strategy_entry = {
            "strategy": strategy,
            "context": context,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        }
        
        self.session_data["strategies_used"].append(strategy_entry)
        
        # Update strategy success rates
        self.strategies_memory["strategy_success_rates"][strategy].append(reward)
        
        # Map context to successful strategies
        context_key = self._context_to_key(context)
        self.strategies_memory["context_strategy_map"][context_key].append({
            "strategy": strategy,
            "reward": reward
        })
        
    def get_best_parameters_for_context(self, context: Dict) -> Optional[Dict[str, Any]]:
        """Retrieve best parameters for similar context."""
        context_key = self._context_to_key(context)
        
        # Look for exact context match
        context_perfs = self.performance_memory["performance_by_context"].get(context_key, [])
        if context_perfs:
            # Return best performing parameters
            best = max(context_perfs, key=lambda x: x["performance"])
            return best["parameters"]
            
        # Find similar contexts
        similar_contexts = self._find_similar_contexts(context)
        if similar_contexts:
            # Average parameters from similar contexts
            param_values = defaultdict(list)
            for ctx_key in similar_contexts:
                for entry in self.performance_memory["performance_by_context"][ctx_key]:
                    for param, value in entry["parameters"].items():
                        param_values[param].append(value)
                        
            # Return averaged parameters
            avg_params = {}
            for param, values in param_values.items():
                if isinstance(values[0], (int, float)):
                    avg_params[param] = np.mean(values)
                else:
                    # For non-numeric, use most common
                    avg_params[param] = max(set(values), key=values.count)
                    
            return avg_params
            
        return None
        
    def get_successful_improvement_patterns(self, improvement_type: str) -> List[Dict]:
        """Get successful patterns for an improvement type."""
        patterns = []
        
        for pattern_key, entries in self.patterns_memory["successful_patterns"].items():
            if improvement_type in pattern_key:
                if entries:
                    # Calculate average success metrics
                    avg_fitness = np.mean([e["fitness"] for e in entries])
                    patterns.append({
                        "pattern": pattern_key,
                        "avg_fitness": avg_fitness,
                        "sample_count": len(entries),
                        "latest": entries[-1]
                    })
                    
        # Sort by average fitness
        patterns.sort(key=lambda x: x["avg_fitness"], reverse=True)
        return patterns[:5]  # Top 5 patterns
        
    def predict_improvement_success(self, improvement: Dict) -> float:
        """Predict likelihood of improvement success based on history."""
        # Base prediction
        base_prob = 0.5
        
        # Check for similar successful improvements
        improvement_key = f"{improvement['type']}_{improvement.get('approach', 'default')}"
        if improvement_key in self.patterns_memory["successful_patterns"]:
            success_history = self.patterns_memory["successful_patterns"][improvement_key]
            if len(success_history) > 3:
                # Calculate success rate
                recent_fitness = [h["fitness"] for h in success_history[-10:]]
                avg_fitness = np.mean(recent_fitness)
                base_prob = min(0.9, avg_fitness)
                
        # Check for failure patterns
        failure_key = f"{improvement['type']}_"
        failure_count = sum(
            count for key, count in self.failures_memory["common_failures"].items()
            if key.startswith(failure_key)
        )
        
        if failure_count > 5:
            base_prob *= 0.7  # Reduce probability if many failures
            
        return base_prob
        
    def get_recommended_strategy(self, context: Dict) -> Optional[str]:
        """Get recommended strategy for given context."""
        context_key = self._context_to_key(context)
        
        # Check direct context mapping
        if context_key in self.strategies_memory["context_strategy_map"]:
            strategies = self.strategies_memory["context_strategy_map"][context_key]
            if strategies:
                # Return best performing strategy
                best = max(strategies, key=lambda x: x["reward"])
                return best["strategy"]
                
        # Check similar contexts
        similar_contexts = self._find_similar_contexts(context)
        strategy_scores = defaultdict(list)
        
        for ctx_key in similar_contexts:
            for entry in self.strategies_memory["context_strategy_map"].get(ctx_key, []):
                strategy_scores[entry["strategy"]].append(entry["reward"])
                
        if strategy_scores:
            # Return strategy with highest average reward
            avg_scores = {
                strat: np.mean(scores) 
                for strat, scores in strategy_scores.items()
            }
            return max(avg_scores, key=avg_scores.get)
            
        return None
        
    def save_session(self):
        """Save current session data."""
        session_file = self.memory_dir / f"session_{self.session_id}.json"
        self.session_data["end_time"] = datetime.now().isoformat()
        
        with open(session_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
            
        # Update long-term memories
        self._update_long_term_memory()
        
    def _update_long_term_memory(self):
        """Update long-term memory components."""
        # Save each memory component
        for memory_type in ["patterns", "performance", "strategies", "failures"]:
            memory_file = self.memory_dir / f"{memory_type}_memory.json"
            memory_data = getattr(self, f"{memory_type}_memory")
            
            # Convert defaultdicts to regular dicts for JSON
            if isinstance(memory_data, dict):
                clean_data = {}
                for key, value in memory_data.items():
                    if isinstance(value, defaultdict):
                        clean_data[key] = dict(value)
                    else:
                        clean_data[key] = value
                        
                with open(memory_file, 'w') as f:
                    json.dump(clean_data, f, indent=2)
                    
    def _get_current_context(self) -> Dict:
        """Get current system context."""
        return {
            "timestamp": datetime.now().isoformat(),
            "session_duration": (
                datetime.now() - datetime.fromisoformat(self.session_data["start_time"])
            ).total_seconds(),
            "improvement_count": len(self.session_data["improvements"]),
            "success_rate": self._calculate_session_success_rate()
        }
        
    def _calculate_session_success_rate(self) -> float:
        """Calculate success rate for current session."""
        if not self.session_data["improvements"]:
            return 0.0
            
        successes = sum(
            1 for imp in self.session_data["improvements"]
            if imp["outcome"].get("success", False)
        )
        return successes / len(self.session_data["improvements"])
        
    def _context_to_key(self, context: Dict) -> str:
        """Convert context dict to hashable key."""
        # Extract key features
        features = [
            f"fit_stag_{int(context.get('fitness_stagnation', 0) * 10)}",
            f"div_{int(context.get('diversity_score', 0.5) * 10)}",
            f"succ_{int(context.get('success_trend', 0.5) * 10)}"
        ]
        return "_".join(features)
        
    def _find_similar_contexts(self, context: Dict, max_results: int = 5) -> List[str]:
        """Find similar contexts in memory."""
        target_key = self._context_to_key(context)
        target_features = self._extract_context_features(context)
        
        similarities = []
        for ctx_key in self.performance_memory["performance_by_context"].keys():
            # Simple similarity based on key matching
            key_similarity = sum(
                1 for part in target_key.split("_")
                if part in ctx_key
            ) / len(target_key.split("_"))
            
            similarities.append((ctx_key, key_similarity))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [ctx_key for ctx_key, _ in similarities[:max_results]]
        
    def _extract_context_features(self, context: Dict) -> np.ndarray:
        """Extract numerical features from context."""
        features = [
            context.get("fitness_stagnation", 0),
            context.get("diversity_score", 0.5),
            context.get("success_trend", 0.5),
            context.get("time_pressure", 0.5),
            context.get("fitness_improvement", 0)
        ]
        return np.array(features)
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of evolutionary memory."""
        return {
            "total_sessions": len(list(self.memory_dir.glob("session_*.json"))),
            "successful_patterns": len(self.patterns_memory["successful_patterns"]),
            "known_failures": sum(self.failures_memory["common_failures"].values()),
            "best_configuration": (
                self.performance_memory["best_configurations"][0]
                if self.performance_memory["best_configurations"] else None
            ),
            "strategy_performance": {
                strat: np.mean(rewards[-10:]) if rewards else 0
                for strat, rewards in self.strategies_memory["strategy_success_rates"].items()
            }
        }