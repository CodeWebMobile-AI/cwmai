"""
Emergent Learning System

Tracks project success and learns what makes projects successful,
adapting validation criteria based on real-world outcomes.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import numpy as np


@dataclass 
class ProjectOutcome:
    """Records the outcome of a created project."""
    project_name: str
    problem_statement: str
    created_at: str
    validation_score: float
    severity_score: float
    market_size: str
    monetization_model: str
    success_metrics: Dict[str, Any]  # Revenue, users, engagement, etc.
    outcome_status: str  # 'successful', 'failed', 'in_progress'
    lessons_learned: List[str]
    

class EmergentLearningSystem:
    """Learns from project outcomes to improve future decisions."""
    
    def __init__(self, learning_file: str = "emergent_learning.json"):
        """Initialize the learning system.
        
        Args:
            learning_file: Path to store learning data
        """
        self.learning_file = learning_file
        self.logger = logging.getLogger(__name__)
        self.outcomes: List[ProjectOutcome] = []
        self.learned_patterns: Dict[str, Any] = {
            'successful_characteristics': [],
            'failure_patterns': [],
            'optimal_thresholds': {
                'validation_score': 7.0,
                'severity_score': 7.0,
                'market_timing_score': 6.0
            },
            'successful_monetization_models': [],
            'successful_problem_types': []
        }
        self.load_learning()
        
    def load_learning(self):
        """Load learning data from file."""
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    data = json.load(f)
                    # Load outcomes
                    self.outcomes = [
                        ProjectOutcome(**outcome) 
                        for outcome in data.get('outcomes', [])
                    ]
                    # Load learned patterns
                    self.learned_patterns = data.get('learned_patterns', self.learned_patterns)
                    self.logger.info(f"Loaded {len(self.outcomes)} project outcomes")
            except Exception as e:
                self.logger.error(f"Error loading learning data: {e}")
                
    def save_learning(self):
        """Save learning data to file."""
        try:
            data = {
                'outcomes': [asdict(outcome) for outcome in self.outcomes],
                'learned_patterns': self.learned_patterns,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.learning_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving learning data: {e}")
            
    def record_outcome(self, project_name: str, problem_statement: str,
                      validation_score: float, severity_score: float,
                      market_size: str, monetization_model: str,
                      success_metrics: Dict[str, Any], 
                      outcome_status: str = 'in_progress'):
        """Record a project outcome.
        
        Args:
            project_name: Name of the project
            problem_statement: The problem it solves
            validation_score: Initial validation score
            severity_score: Initial severity score
            market_size: Estimated market size
            monetization_model: How it makes money
            success_metrics: Metrics indicating success
            outcome_status: Current status
        """
        outcome = ProjectOutcome(
            project_name=project_name,
            problem_statement=problem_statement,
            created_at=datetime.now().isoformat(),
            validation_score=validation_score,
            severity_score=severity_score,
            market_size=market_size,
            monetization_model=monetization_model,
            success_metrics=success_metrics,
            outcome_status=outcome_status,
            lessons_learned=[]
        )
        
        self.outcomes.append(outcome)
        self.analyze_patterns()
        self.save_learning()
        
    def update_outcome(self, project_name: str, success_metrics: Dict[str, Any],
                      outcome_status: str, lessons: List[str]):
        """Update an existing project outcome.
        
        Args:
            project_name: Name of the project
            success_metrics: Updated metrics
            outcome_status: New status
            lessons: Lessons learned
        """
        for outcome in self.outcomes:
            if outcome.project_name == project_name:
                outcome.success_metrics.update(success_metrics)
                outcome.outcome_status = outcome_status
                outcome.lessons_learned.extend(lessons)
                break
                
        self.analyze_patterns()
        self.save_learning()
        
    def analyze_patterns(self):
        """Analyze outcomes to learn patterns."""
        if len(self.outcomes) < 3:
            return  # Need more data
            
        successful = [o for o in self.outcomes if o.outcome_status == 'successful']
        failed = [o for o in self.outcomes if o.outcome_status == 'failed']
        
        if successful:
            # Learn from successful projects
            avg_validation = np.mean([o.validation_score for o in successful])
            avg_severity = np.mean([o.severity_score for o in successful])
            
            # Update optimal thresholds based on success
            self.learned_patterns['optimal_thresholds']['validation_score'] = max(6.0, avg_validation - 1)
            self.learned_patterns['optimal_thresholds']['severity_score'] = max(6.0, avg_severity - 1)
            
            # Track successful monetization models
            monetization_models = [o.monetization_model for o in successful]
            self.learned_patterns['successful_monetization_models'] = list(set(monetization_models))
            
            # Extract common characteristics
            self.learned_patterns['successful_characteristics'] = self._extract_characteristics(successful)
            
        if failed:
            # Learn from failures
            self.learned_patterns['failure_patterns'] = self._extract_characteristics(failed)
            
    def _extract_characteristics(self, outcomes: List[ProjectOutcome]) -> List[str]:
        """Extract common characteristics from outcomes.
        
        Args:
            outcomes: List of project outcomes
            
        Returns:
            List of common characteristics
        """
        characteristics = []
        
        # Analyze problem statements for patterns
        problem_words = {}
        for outcome in outcomes:
            words = outcome.problem_statement.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    problem_words[word] = problem_words.get(word, 0) + 1
                    
        # Find common words
        common_words = [
            word for word, count in problem_words.items() 
            if count >= len(outcomes) * 0.5  # Appears in 50%+ of outcomes
        ]
        
        if common_words:
            characteristics.append(f"Common problem themes: {', '.join(common_words[:5])}")
            
        # Analyze scores
        avg_validation = np.mean([o.validation_score for o in outcomes])
        avg_severity = np.mean([o.severity_score for o in outcomes])
        characteristics.append(f"Average validation score: {avg_validation:.1f}")
        characteristics.append(f"Average severity score: {avg_severity:.1f}")
        
        return characteristics
        
    def get_adjusted_thresholds(self) -> Dict[str, float]:
        """Get adjusted thresholds based on learning.
        
        Returns:
            Dictionary of adjusted thresholds
        """
        return self.learned_patterns['optimal_thresholds']
        
    def should_create_project(self, validation_score: float, severity_score: float,
                            monetization_model: str) -> Tuple[bool, str]:
        """Determine if a project should be created based on learning.
        
        Args:
            validation_score: The validation score
            severity_score: The severity score
            monetization_model: Proposed monetization model
            
        Returns:
            Tuple of (should_create, reason)
        """
        thresholds = self.get_adjusted_thresholds()
        
        # Check against learned thresholds
        if validation_score < thresholds['validation_score']:
            return False, f"Validation score {validation_score} below learned threshold {thresholds['validation_score']}"
            
        if severity_score < thresholds['severity_score']:
            return False, f"Severity score {severity_score} below learned threshold {thresholds['severity_score']}"
            
        # Bonus for proven monetization models
        if monetization_model in self.learned_patterns.get('successful_monetization_models', []):
            return True, f"Proven monetization model: {monetization_model}"
            
        return True, "Meets all criteria"
        
    def get_insights(self) -> Dict[str, Any]:
        """Get insights from the learning system.
        
        Returns:
            Dictionary of insights
        """
        total_projects = len(self.outcomes)
        successful = len([o for o in self.outcomes if o.outcome_status == 'successful'])
        failed = len([o for o in self.outcomes if o.outcome_status == 'failed'])
        in_progress = len([o for o in self.outcomes if o.outcome_status == 'in_progress'])
        
        success_rate = successful / total_projects if total_projects > 0 else 0
        
        return {
            'total_projects': total_projects,
            'successful': successful,
            'failed': failed,
            'in_progress': in_progress,
            'success_rate': success_rate,
            'optimal_thresholds': self.learned_patterns['optimal_thresholds'],
            'successful_monetization_models': self.learned_patterns['successful_monetization_models'],
            'successful_characteristics': self.learned_patterns['successful_characteristics'],
            'failure_patterns': self.learned_patterns['failure_patterns']
        }
        
    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on learning.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        insights = self.get_insights()
        
        if insights['success_rate'] < 0.5 and insights['total_projects'] >= 5:
            suggestions.append("Consider lowering validation thresholds - current success rate is low")
            
        if insights['failure_patterns']:
            suggestions.append(f"Avoid these patterns: {', '.join(insights['failure_patterns'][:3])}")
            
        if insights['successful_monetization_models']:
            suggestions.append(f"Focus on these proven models: {', '.join(insights['successful_monetization_models'])}")
            
        if not suggestions:
            suggestions.append("Continue with current approach - metrics look good")
            
        return suggestions