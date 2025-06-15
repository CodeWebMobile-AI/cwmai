"""
Improvement Learning System

Learns from successful and failed improvements to enhance future suggestions.
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
from dataclasses import dataclass, asdict

from safe_self_improver import ModificationType
from ai_code_analyzer import CodeImprovement


@dataclass
class ImprovementOutcome:
    """Records the outcome of an applied improvement."""
    improvement_id: str
    improvement_type: ModificationType
    file_path: str
    pattern_signature: str
    success: bool
    performance_impact: float  # -1.0 to 1.0
    error_reduction: float  # 0.0 to 1.0
    readability_impact: float  # -1.0 to 1.0
    timestamp: datetime
    feedback: Optional[str] = None
    context: Dict[str, Any] = None


class ImprovementLearningSystem:
    """Learns from improvement outcomes to enhance future suggestions."""
    
    def __init__(self, learning_dir: str = ".self_improver/learning"):
        """Initialize learning system.
        
        Args:
            learning_dir: Directory to store learning data
        """
        self.learning_dir = learning_dir
        os.makedirs(learning_dir, exist_ok=True)
        
        self.outcomes_file = os.path.join(learning_dir, "outcomes.json")
        self.patterns_file = os.path.join(learning_dir, "patterns.pkl")
        self.weights_file = os.path.join(learning_dir, "weights.json")
        
        self.outcomes = self._load_outcomes()
        self.pattern_success = self._load_patterns()
        self.feature_weights = self._load_weights()
        
    def _load_outcomes(self) -> List[ImprovementOutcome]:
        """Load historical improvement outcomes."""
        if not os.path.exists(self.outcomes_file):
            return []
        
        try:
            with open(self.outcomes_file, 'r') as f:
                data = json.load(f)
                outcomes = []
                for item in data:
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    # Handle enum by name or value
                    imp_type = item['improvement_type']
                    if isinstance(imp_type, str):
                        try:
                            item['improvement_type'] = ModificationType[imp_type.upper()]
                        except KeyError:
                            # Try to match by value
                            for mt in ModificationType:
                                if mt.value == imp_type:
                                    item['improvement_type'] = mt
                                    break
                    else:
                        item['improvement_type'] = ModificationType(imp_type)
                    outcomes.append(ImprovementOutcome(**item))
                return outcomes
        except Exception as e:
            print(f"Error loading outcomes: {e}")
            return []
    
    def _load_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load learned pattern success rates."""
        if not os.path.exists(self.patterns_file):
            return defaultdict(lambda: {'success_rate': 0.5, 'count': 0})
        
        try:
            with open(self.patterns_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading patterns: {e}")
            return defaultdict(lambda: {'success_rate': 0.5, 'count': 0})
    
    def _load_weights(self) -> Dict[str, float]:
        """Load feature weights for scoring improvements."""
        if not os.path.exists(self.weights_file):
            return {
                'confidence': 1.0,
                'pattern_success': 1.5,
                'complexity_reduction': 1.2,
                'performance_gain': 1.3,
                'risk_level': -0.8,
                'file_criticality': 1.1
            }
        
        try:
            with open(self.weights_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading weights: {e}")
            return self._load_weights()  # Return defaults
    
    def record_outcome(self, improvement: CodeImprovement, success: bool,
                      metrics: Optional[Dict[str, float]] = None,
                      feedback: Optional[str] = None):
        """Record the outcome of an applied improvement.
        
        Args:
            improvement: The improvement that was applied
            success: Whether the improvement was successful
            metrics: Performance/quality metrics
            feedback: Optional human feedback
        """
        pattern_sig = self._generate_pattern_signature(improvement)
        
        outcome = ImprovementOutcome(
            improvement_id=f"{improvement.type.value}_{hash(improvement.description)}",
            improvement_type=improvement.type,
            file_path=improvement.original_code[:50],  # Store partial for privacy
            pattern_signature=pattern_sig,
            success=success,
            performance_impact=metrics.get('performance', 0.0) if metrics else 0.0,
            error_reduction=metrics.get('error_reduction', 0.0) if metrics else 0.0,
            readability_impact=metrics.get('readability', 0.0) if metrics else 0.0,
            timestamp=datetime.now(),
            feedback=feedback,
            context=improvement.impact_analysis
        )
        
        self.outcomes.append(outcome)
        self._update_pattern_success(pattern_sig, success)
        self._save_outcomes()
        
        # Periodically update weights based on outcomes
        if len(self.outcomes) % 10 == 0:
            self._update_feature_weights()
    
    def _generate_pattern_signature(self, improvement: CodeImprovement) -> str:
        """Generate a signature for the improvement pattern."""
        # Create a signature that captures the essence of the improvement
        signature_parts = [
            improvement.type.value,
            str(len(improvement.original_code.split('\n'))),
            'has_loop' if 'for' in improvement.original_code else 'no_loop',
            'has_if' if 'if' in improvement.original_code else 'no_if',
            'has_function' if 'def' in improvement.original_code else 'no_function'
        ]
        
        return '_'.join(signature_parts)
    
    def _update_pattern_success(self, pattern_sig: str, success: bool):
        """Update success rate for a pattern."""
        if pattern_sig not in self.pattern_success:
            self.pattern_success[pattern_sig] = {'success_rate': 0.5, 'count': 0}
        
        pattern_data = self.pattern_success[pattern_sig]
        count = pattern_data['count']
        current_rate = pattern_data['success_rate']
        
        # Update with exponential moving average
        alpha = 0.2  # Learning rate
        new_rate = (1 - alpha) * current_rate + alpha * (1.0 if success else 0.0)
        
        pattern_data['success_rate'] = new_rate
        pattern_data['count'] = count + 1
        
        self._save_patterns()
    
    def score_improvement(self, improvement: CodeImprovement,
                         context: Optional[Dict[str, Any]] = None) -> float:
        """Score an improvement based on learned patterns and features.
        
        Args:
            improvement: The improvement to score
            context: Additional context (file type, project info, etc.)
            
        Returns:
            Score from 0.0 to 1.0
        """
        features = self._extract_features(improvement, context)
        
        # Apply learned weights
        score = 0.0
        for feature, value in features.items():
            weight = self.feature_weights.get(feature, 1.0)
            score += weight * value
        
        # Normalize to 0-1 range
        score = max(0.0, min(1.0, score / sum(abs(w) for w in self.feature_weights.values())))
        
        # Boost or penalize based on pattern history
        pattern_sig = self._generate_pattern_signature(improvement)
        if pattern_sig in self.pattern_success:
            pattern_data = self.pattern_success[pattern_sig]
            if pattern_data['count'] >= 3:  # Enough data
                score = 0.7 * score + 0.3 * pattern_data['success_rate']
        
        return score
    
    def _extract_features(self, improvement: CodeImprovement,
                         context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Extract features for scoring."""
        features = {
            'confidence': improvement.confidence,
            'complexity_reduction': self._estimate_complexity_reduction(improvement),
            'performance_gain': self._estimate_performance_gain(improvement),
            'risk_level': self._assess_risk(improvement),
            'file_criticality': self._assess_file_criticality(context)
        }
        
        # Add pattern success rate if available
        pattern_sig = self._generate_pattern_signature(improvement)
        if pattern_sig in self.pattern_success:
            features['pattern_success'] = self.pattern_success[pattern_sig]['success_rate']
        else:
            features['pattern_success'] = 0.5  # Neutral
        
        return features
    
    def _estimate_complexity_reduction(self, improvement: CodeImprovement) -> float:
        """Estimate how much the improvement reduces complexity."""
        original_lines = len(improvement.original_code.split('\n'))
        improved_lines = len(improvement.improved_code.split('\n'))
        
        # Fewer lines is generally better
        line_reduction = (original_lines - improved_lines) / max(original_lines, 1)
        
        # Check for common complexity reducers
        complexity_score = 0.0
        
        if 'comprehension' in improvement.description.lower():
            complexity_score += 0.3
        if 'enumerate' in improvement.improved_code and 'range(len' in improvement.original_code:
            complexity_score += 0.2
        if improvement.type == ModificationType.REFACTORING:
            complexity_score += 0.1
        
        return min(1.0, line_reduction * 0.5 + complexity_score)
    
    def _estimate_performance_gain(self, improvement: CodeImprovement) -> float:
        """Estimate potential performance gain."""
        perf_score = 0.0
        
        # Check improvement type
        if improvement.type == ModificationType.OPTIMIZATION:
            perf_score += 0.3
        
        # Check for known performance patterns
        if 'comprehension' in improvement.description.lower():
            perf_score += 0.2
        if 'cache' in improvement.description.lower():
            perf_score += 0.3
        if 'vectorize' in improvement.description.lower():
            perf_score += 0.4
        
        # Check impact analysis
        if improvement.impact_analysis.get('performance') == 'high':
            perf_score += 0.3
        elif improvement.impact_analysis.get('performance') == 'medium':
            perf_score += 0.1
        
        return min(1.0, perf_score)
    
    def _assess_risk(self, improvement: CodeImprovement) -> float:
        """Assess risk level of the improvement."""
        risk = 0.0
        
        # Type-based risk
        risk_by_type = {
            ModificationType.DOCUMENTATION: 0.1,
            ModificationType.OPTIMIZATION: 0.3,
            ModificationType.REFACTORING: 0.5,
            ModificationType.SECURITY: 0.7,
            ModificationType.FEATURE_ADDITION: 0.8
        }
        risk += risk_by_type.get(improvement.type, 0.5)
        
        # Scope-based risk
        lines_changed = improvement.line_end - improvement.line_start
        if lines_changed > 50:
            risk += 0.3
        elif lines_changed > 20:
            risk += 0.1
        
        # Confidence adjustment
        risk *= (2.0 - improvement.confidence)
        
        return min(1.0, risk)
    
    def _assess_file_criticality(self, context: Optional[Dict[str, Any]]) -> float:
        """Assess how critical the file is."""
        if not context:
            return 0.5
        
        criticality = 0.5
        
        file_type = context.get('file_type', 'general')
        critical_types = {'api': 0.8, 'auth': 0.9, 'security': 0.9, 'database': 0.8}
        
        if file_type in critical_types:
            criticality = critical_types[file_type]
        elif file_type == 'test':
            criticality = 0.3
        elif file_type == 'utility':
            criticality = 0.4
        
        return criticality
    
    def get_recommendations(self, improvement_type: Optional[ModificationType] = None,
                           file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recommendations based on learned patterns.
        
        Args:
            improvement_type: Filter by improvement type
            file_type: Filter by file type
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze successful patterns
        for pattern_sig, pattern_data in self.pattern_success.items():
            if pattern_data['count'] < 5:
                continue
            
            parts = pattern_sig.split('_')
            # Handle case where the enum value is stored, not the name
            try:
                pattern_type = ModificationType[parts[0].upper()]
            except KeyError:
                # Try to match by value
                for mt in ModificationType:
                    if mt.value == parts[0]:
                        pattern_type = mt
                        break
                else:
                    continue
            
            if improvement_type and pattern_type != improvement_type:
                continue
            
            if pattern_data['success_rate'] > 0.7:
                recommendations.append({
                    'pattern': pattern_sig,
                    'type': pattern_type.value,
                    'success_rate': pattern_data['success_rate'],
                    'count': pattern_data['count'],
                    'recommendation': f"This pattern has {pattern_data['success_rate']:.0%} success rate"
                })
        
        # Sort by success rate
        recommendations.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return recommendations[:10]  # Top 10
    
    def _update_feature_weights(self):
        """Update feature weights based on outcomes."""
        # Simple gradient update based on outcome correlation
        feature_impacts = defaultdict(list)
        
        for outcome in self.outcomes[-50:]:  # Last 50 outcomes
            # Extract features from context
            features = {
                'confidence': outcome.context.get('confidence', 0.5),
                'pattern_success': self.pattern_success.get(
                    outcome.pattern_signature, {}
                ).get('success_rate', 0.5),
                'complexity_reduction': outcome.context.get('complexity_reduction', 0.0),
                'performance_gain': outcome.performance_impact,
                'risk_level': outcome.context.get('risk_level', 0.5),
                'file_criticality': outcome.context.get('file_criticality', 0.5)
            }
            
            # Record impact
            for feature, value in features.items():
                impact = value * (1.0 if outcome.success else -1.0)
                feature_impacts[feature].append(impact)
        
        # Update weights
        learning_rate = 0.1
        for feature, impacts in feature_impacts.items():
            if impacts:
                avg_impact = sum(impacts) / len(impacts)
                current_weight = self.feature_weights.get(feature, 1.0)
                new_weight = current_weight + learning_rate * avg_impact
                self.feature_weights[feature] = max(-2.0, min(2.0, new_weight))
        
        self._save_weights()
    
    def _save_outcomes(self):
        """Save outcomes to file."""
        data = []
        for outcome in self.outcomes:
            item = asdict(outcome)
            item['timestamp'] = item['timestamp'].isoformat()
            item['improvement_type'] = item['improvement_type'].value
            data.append(item)
        
        with open(self.outcomes_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_patterns(self):
        """Save pattern data."""
        with open(self.patterns_file, 'wb') as f:
            pickle.dump(dict(self.pattern_success), f)
    
    def _save_weights(self):
        """Save feature weights."""
        with open(self.weights_file, 'w') as f:
            json.dump(self.feature_weights, f, indent=2)
    
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate a report on learned patterns and performance."""
        total_outcomes = len(self.outcomes)
        successful = sum(1 for o in self.outcomes if o.success)
        
        # Calculate success rates by type
        by_type = defaultdict(lambda: {'total': 0, 'successful': 0})
        for outcome in self.outcomes:
            type_key = outcome.improvement_type.value
            by_type[type_key]['total'] += 1
            if outcome.success:
                by_type[type_key]['successful'] += 1
        
        # Find best patterns
        best_patterns = []
        for pattern_sig, pattern_data in self.pattern_success.items():
            if pattern_data['count'] >= 5:
                best_patterns.append({
                    'pattern': pattern_sig,
                    'success_rate': pattern_data['success_rate'],
                    'count': pattern_data['count']
                })
        best_patterns.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return {
            'summary': {
                'total_improvements': total_outcomes,
                'successful_improvements': successful,
                'overall_success_rate': successful / total_outcomes if total_outcomes > 0 else 0,
                'patterns_learned': len(self.pattern_success)
            },
            'by_type': {
                k: {
                    'total': v['total'],
                    'successful': v['successful'],
                    'success_rate': v['successful'] / v['total'] if v['total'] > 0 else 0
                }
                for k, v in by_type.items()
            },
            'feature_weights': self.feature_weights,
            'best_patterns': best_patterns[:5],
            'recent_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> Dict[str, float]:
        """Calculate recent success trend."""
        if len(self.outcomes) < 10:
            return {'trend': 'insufficient_data', 'recent_rate': 0.0, 'overall_rate': 0.0}
        
        recent = self.outcomes[-10:]
        recent_success = sum(1 for o in recent if o.success) / len(recent)
        
        all_time = self.outcomes[:-10]
        overall_success = sum(1 for o in all_time if o.success) / len(all_time) if all_time else 0
        
        return {
            'trend': 'improving' if recent_success > overall_success else 'declining',
            'recent_rate': recent_success,
            'overall_rate': overall_success
        }