"""
Auto-implement evolution recommendations
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

class AutoRecommendationHandler:
    """Automatically apply recommendations from the evolution system."""
    
    def __init__(self, evolution_runner):
        self.runner = evolution_runner
        self.applied_recommendations = []
        self.recommendation_history = []
        self.logger = logging.getLogger(__name__)
        
        # Track parameter bounds
        self.parameter_bounds = {
            'mutation_rate': {'min': 0.05, 'max': 0.5, 'step': 0.05},
            'population_size': {'min': 5, 'max': 30, 'step': 2},
            'synthesis_frequency': {'min': 3, 'max': 20, 'step': 2}
        }
        
    def apply_recommendations(self, recommendations: List[str]) -> Dict[str, Any]:
        """Parse and apply recommendations automatically."""
        
        applied_changes = {}
        
        for rec in recommendations:
            rec_lower = rec.lower()
            
            # Skip if already applied
            if any(rec in applied for applied in self.applied_recommendations):
                continue
                
            # Mutation rate adjustments
            if "mutation rate" in rec_lower:
                old_rate = getattr(self.runner.evolutionary_engine, 'mutation_rate', 0.1)
                
                if "increas" in rec_lower:
                    # Increase by 20%
                    new_rate = min(
                        self.parameter_bounds['mutation_rate']['max'],
                        old_rate * 1.2
                    )
                elif "decreas" in rec_lower:
                    # Decrease by 20%
                    new_rate = max(
                        self.parameter_bounds['mutation_rate']['min'],
                        old_rate * 0.8
                    )
                else:
                    continue
                    
                if hasattr(self.runner.evolutionary_engine, 'mutation_rate'):
                    self.runner.evolutionary_engine.mutation_rate = new_rate
                    applied_changes['mutation_rate'] = {'old': old_rate, 'new': new_rate}
                    print(f"🔧 Auto-adjusted mutation rate: {old_rate:.2f} → {new_rate:.2f}")
                    self.applied_recommendations.append(rec)
                    
            # Population size adjustments
            elif "population size" in rec_lower:
                old_size = getattr(self.runner.evolutionary_engine, 'population_size', 10)
                
                if "increas" in rec_lower:
                    # Increase by step size
                    new_size = min(
                        self.parameter_bounds['population_size']['max'],
                        old_size + self.parameter_bounds['population_size']['step']
                    )
                elif "decreas" in rec_lower:
                    # Decrease by step size
                    new_size = max(
                        self.parameter_bounds['population_size']['min'],
                        old_size - self.parameter_bounds['population_size']['step']
                    )
                else:
                    continue
                    
                if hasattr(self.runner.evolutionary_engine, 'population_size'):
                    self.runner.evolutionary_engine.population_size = new_size
                    applied_changes['population_size'] = {'old': old_size, 'new': new_size}
                    print(f"🔧 Auto-adjusted population size: {old_size} → {new_size}")
                    self.applied_recommendations.append(rec)
                    
            # Synthesis frequency adjustments
            elif "synthesis frequency" in rec_lower:
                old_freq = self.runner.auto_tuning_state.get('synthesis_frequency', 10)
                
                if "increas" in rec_lower:
                    # Decrease interval (more frequent)
                    new_freq = max(
                        self.parameter_bounds['synthesis_frequency']['min'],
                        old_freq - self.parameter_bounds['synthesis_frequency']['step']
                    )
                elif "decreas" in rec_lower:
                    # Increase interval (less frequent)
                    new_freq = min(
                        self.parameter_bounds['synthesis_frequency']['max'],
                        old_freq + self.parameter_bounds['synthesis_frequency']['step']
                    )
                else:
                    continue
                    
                # Update the synthesis frequency in auto-tuning state
                self.runner.auto_tuning_state['synthesis_frequency'] = new_freq
                applied_changes['synthesis_frequency'] = {'old': old_freq, 'new': new_freq}
                print(f"🔧 Auto-adjusted synthesis frequency: every {old_freq} → every {new_freq} generations")
                self.applied_recommendations.append(rec)
                
            # Learning rate adjustments (if applicable)
            elif "learning rate" in rec_lower:
                if hasattr(self.runner.evolutionary_engine, 'learning_rate'):
                    old_lr = self.runner.evolutionary_engine.learning_rate
                    if "increas" in rec_lower:
                        new_lr = min(0.1, old_lr * 1.1)
                    elif "decreas" in rec_lower:
                        new_lr = max(0.001, old_lr * 0.9)
                    else:
                        continue
                        
                    self.runner.evolutionary_engine.learning_rate = new_lr
                    applied_changes['learning_rate'] = {'old': old_lr, 'new': new_lr}
                    print(f"🔧 Auto-adjusted learning rate: {old_lr:.4f} → {new_lr:.4f}")
                    self.applied_recommendations.append(rec)
        
        # Record this tuning event
        if applied_changes:
            self.recommendation_history.append({
                'timestamp': datetime.now().isoformat(),
                'generation': self.runner.generations_completed,
                'recommendations': recommendations,
                'applied_changes': applied_changes
            })
            
        return applied_changes
    
    def get_tuning_history(self) -> List[Dict[str, Any]]:
        """Get the history of all tuning events."""
        return self.recommendation_history
    
    def reset_recommendations(self):
        """Reset the applied recommendations list to allow re-application."""
        self.applied_recommendations = []
        print("🔄 Reset auto-tuning recommendations")