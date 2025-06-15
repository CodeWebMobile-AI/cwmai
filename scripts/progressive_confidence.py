"""
Progressive Confidence System

Builds confidence in auto-applying improvements based on historical success.
Starts conservatively and gradually increases automation.
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

from safe_self_improver import ModificationType


class RiskLevel(Enum):
    """Risk levels for improvements."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HistoricalOutcome:
    """Historical outcome of an improvement."""
    staging_id: str
    improvement_type: ModificationType
    risk_level: RiskLevel
    applied_at: datetime
    success: bool
    had_issues: bool
    rollback_required: bool
    performance_impact: float  # -1 to 1, negative is bad
    error_rate_change: float  # -1 to 1, negative is bad
    user_feedback: Optional[str] = None


@dataclass
class ConfidenceMetrics:
    """Metrics for confidence calculation."""
    total_improvements: int = 0
    successful_improvements: int = 0
    failed_improvements: int = 0
    rollbacks_required: int = 0
    days_since_start: int = 0
    success_rate: float = 0.0
    success_rate_by_type: Dict[str, float] = field(default_factory=dict)
    success_rate_by_risk: Dict[str, float] = field(default_factory=dict)
    recent_success_rate: float = 0.0  # Last 30 days
    confidence_score: float = 0.0


class ProgressiveConfidence:
    """System that builds confidence in improvements over time."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize confidence system."""
        self.repo_path = os.path.abspath(repo_path)
        self.history_file = os.path.join(
            repo_path, '.self_improver', 'confidence_history.json'
        )
        self.config_file = os.path.join(
            repo_path, '.self_improver', 'confidence_config.json'
        )
        
        # Load history and config
        self.history: List[HistoricalOutcome] = self._load_history()
        self.config = self._load_config()
        
        # Calculate current metrics
        self.metrics = self._calculate_metrics()
    
    def _load_history(self) -> List[HistoricalOutcome]:
        """Load historical outcomes."""
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
            
            history = []
            for item in data:
                history.append(HistoricalOutcome(
                    staging_id=item['staging_id'],
                    improvement_type=ModificationType(item['improvement_type']),
                    risk_level=RiskLevel(item['risk_level']),
                    applied_at=datetime.fromisoformat(item['applied_at']),
                    success=item['success'],
                    had_issues=item['had_issues'],
                    rollback_required=item['rollback_required'],
                    performance_impact=item['performance_impact'],
                    error_rate_change=item['error_rate_change'],
                    user_feedback=item.get('user_feedback')
                ))
            
            return history
            
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load confidence configuration."""
        default_config = {
            # Time-based thresholds (days)
            'initial_manual_period_days': 7,
            'cautious_period_days': 30,
            'confident_period_days': 90,
            
            # Success rate thresholds
            'min_success_rate_for_auto': 0.95,
            'min_success_rate_by_type': 0.90,
            'min_recent_success_rate': 0.98,
            
            # Sample size requirements
            'min_improvements_for_auto': 10,
            'min_improvements_per_type': 3,
            
            # Risk-based rules
            'max_auto_risk_initial': 'low',
            'max_auto_risk_cautious': 'medium',
            'max_auto_risk_confident': 'high',
            
            # Confidence score thresholds
            'min_confidence_for_low_risk': 0.7,
            'min_confidence_for_medium_risk': 0.85,
            'min_confidence_for_high_risk': 0.95,
            
            # Cooldown periods
            'failure_cooldown_days': 3,
            'rollback_cooldown_days': 7
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
            except:
                pass
        
        # Save config
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _calculate_metrics(self) -> ConfidenceMetrics:
        """Calculate current confidence metrics."""
        metrics = ConfidenceMetrics()
        
        if not self.history:
            return metrics
        
        # Basic counts
        metrics.total_improvements = len(self.history)
        metrics.successful_improvements = sum(1 for h in self.history if h.success)
        metrics.failed_improvements = sum(1 for h in self.history if not h.success)
        metrics.rollbacks_required = sum(1 for h in self.history if h.rollback_required)
        
        # Days since start
        first_improvement = min(self.history, key=lambda h: h.applied_at)
        metrics.days_since_start = (datetime.now(timezone.utc) - first_improvement.applied_at).days
        
        # Overall success rate
        metrics.success_rate = (
            metrics.successful_improvements / metrics.total_improvements
            if metrics.total_improvements > 0 else 0
        )
        
        # Success rate by type
        by_type = {}
        for imp_type in ModificationType:
            type_improvements = [h for h in self.history if h.improvement_type == imp_type]
            if type_improvements:
                success_count = sum(1 for h in type_improvements if h.success)
                by_type[imp_type.value] = success_count / len(type_improvements)
        metrics.success_rate_by_type = by_type
        
        # Success rate by risk
        by_risk = {}
        for risk in RiskLevel:
            risk_improvements = [h for h in self.history if h.risk_level == risk]
            if risk_improvements:
                success_count = sum(1 for h in risk_improvements if h.success)
                by_risk[risk.value] = success_count / len(risk_improvements)
        metrics.success_rate_by_risk = by_risk
        
        # Recent success rate (last 30 days)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        recent_improvements = [h for h in self.history if h.applied_at > recent_cutoff]
        if recent_improvements:
            recent_success = sum(1 for h in recent_improvements if h.success)
            metrics.recent_success_rate = recent_success / len(recent_improvements)
        
        # Calculate confidence score
        metrics.confidence_score = self._calculate_confidence_score(metrics)
        
        return metrics
    
    def _calculate_confidence_score(self, metrics: ConfidenceMetrics) -> float:
        """Calculate overall confidence score (0-1)."""
        factors = []
        weights = []
        
        # Success rate factor (weight: 30%)
        factors.append(metrics.success_rate)
        weights.append(0.3)
        
        # Recent success rate factor (weight: 25%)
        factors.append(metrics.recent_success_rate)
        weights.append(0.25)
        
        # Experience factor (weight: 20%)
        experience_score = min(1.0, metrics.total_improvements / 50)
        factors.append(experience_score)
        weights.append(0.2)
        
        # Time factor (weight: 15%)
        time_score = min(1.0, metrics.days_since_start / 90)
        factors.append(time_score)
        weights.append(0.15)
        
        # Stability factor (weight: 10%)
        stability_score = 1.0 - (metrics.rollbacks_required / max(metrics.total_improvements, 1))
        factors.append(stability_score)
        weights.append(0.1)
        
        # Calculate weighted average
        confidence = sum(f * w for f, w in zip(factors, weights))
        
        return min(1.0, max(0.0, confidence))
    
    def should_auto_apply(self, improvement_type: ModificationType, 
                         risk_level: RiskLevel) -> Tuple[bool, str]:
        """Determine if an improvement should be auto-applied.
        
        Args:
            improvement_type: Type of improvement
            risk_level: Risk level of improvement
            
        Returns:
            Tuple of (should_apply, reason)
        """
        # Check if in cooldown
        cooldown_reason = self._check_cooldown()
        if cooldown_reason:
            return False, cooldown_reason
        
        # Initial manual period
        if self.metrics.days_since_start < self.config['initial_manual_period_days']:
            return False, f"Still in initial manual period ({self.metrics.days_since_start}/{self.config['initial_manual_period_days']} days)"
        
        # Check minimum improvements
        if self.metrics.total_improvements < self.config['min_improvements_for_auto']:
            return False, f"Insufficient history ({self.metrics.total_improvements}/{self.config['min_improvements_for_auto']} improvements)"
        
        # Check overall success rate
        if self.metrics.success_rate < self.config['min_success_rate_for_auto']:
            return False, f"Success rate too low ({self.metrics.success_rate:.1%} < {self.config['min_success_rate_for_auto']:.1%})"
        
        # Check recent success rate
        if self.metrics.recent_success_rate < self.config['min_recent_success_rate']:
            return False, f"Recent success rate too low ({self.metrics.recent_success_rate:.1%} < {self.config['min_recent_success_rate']:.1%})"
        
        # Check type-specific success rate
        type_success_rate = self.metrics.success_rate_by_type.get(improvement_type.value, 0)
        if type_success_rate < self.config['min_success_rate_by_type']:
            return False, f"{improvement_type.value} success rate too low ({type_success_rate:.1%})"
        
        # Check risk level based on confidence period
        max_risk = self._get_max_auto_risk()
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        if risk_order.index(risk_level) > risk_order.index(RiskLevel(max_risk)):
            return False, f"Risk level {risk_level.value} exceeds current maximum ({max_risk})"
        
        # Check confidence score for risk level
        required_confidence = {
            RiskLevel.LOW: self.config['min_confidence_for_low_risk'],
            RiskLevel.MEDIUM: self.config['min_confidence_for_medium_risk'],
            RiskLevel.HIGH: self.config['min_confidence_for_high_risk'],
            RiskLevel.CRITICAL: 1.0  # Never auto-apply critical
        }
        
        if self.metrics.confidence_score < required_confidence.get(risk_level, 1.0):
            return False, f"Confidence too low for {risk_level.value} risk ({self.metrics.confidence_score:.2f} < {required_confidence[risk_level]:.2f})"
        
        # All checks passed
        return True, "All auto-apply criteria met"
    
    def _check_cooldown(self) -> Optional[str]:
        """Check if in cooldown period."""
        if not self.history:
            return None
        
        now = datetime.now(timezone.utc)
        
        # Check failure cooldown
        recent_failures = [
            h for h in self.history 
            if not h.success and 
            (now - h.applied_at).days < self.config['failure_cooldown_days']
        ]
        if recent_failures:
            days_remaining = self.config['failure_cooldown_days'] - (now - recent_failures[-1].applied_at).days
            return f"In failure cooldown ({days_remaining} days remaining)"
        
        # Check rollback cooldown
        recent_rollbacks = [
            h for h in self.history 
            if h.rollback_required and 
            (now - h.applied_at).days < self.config['rollback_cooldown_days']
        ]
        if recent_rollbacks:
            days_remaining = self.config['rollback_cooldown_days'] - (now - recent_rollbacks[-1].applied_at).days
            return f"In rollback cooldown ({days_remaining} days remaining)"
        
        return None
    
    def _get_max_auto_risk(self) -> str:
        """Get maximum auto-apply risk level based on confidence period."""
        days = self.metrics.days_since_start
        
        if days < self.config['cautious_period_days']:
            return self.config['max_auto_risk_initial']
        elif days < self.config['confident_period_days']:
            return self.config['max_auto_risk_cautious']
        else:
            return self.config['max_auto_risk_confident']
    
    def record_outcome(self, staging_id: str, improvement_type: ModificationType,
                      risk_level: RiskLevel, success: bool, 
                      had_issues: bool = False, rollback_required: bool = False,
                      performance_impact: float = 0.0, 
                      error_rate_change: float = 0.0,
                      user_feedback: Optional[str] = None):
        """Record the outcome of an improvement."""
        outcome = HistoricalOutcome(
            staging_id=staging_id,
            improvement_type=improvement_type,
            risk_level=risk_level,
            applied_at=datetime.now(timezone.utc),
            success=success,
            had_issues=had_issues,
            rollback_required=rollback_required,
            performance_impact=performance_impact,
            error_rate_change=error_rate_change,
            user_feedback=user_feedback
        )
        
        self.history.append(outcome)
        self._save_history()
        
        # Recalculate metrics
        self.metrics = self._calculate_metrics()
        
        # Log outcome
        print(f"📊 Recorded outcome: {improvement_type.value} - {'✅ Success' if success else '❌ Failed'}")
    
    def _save_history(self):
        """Save history to file."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        data = []
        for outcome in self.history:
            data.append({
                'staging_id': outcome.staging_id,
                'improvement_type': outcome.improvement_type.value,
                'risk_level': outcome.risk_level.value,
                'applied_at': outcome.applied_at.isoformat(),
                'success': outcome.success,
                'had_issues': outcome.had_issues,
                'rollback_required': outcome.rollback_required,
                'performance_impact': outcome.performance_impact,
                'error_rate_change': outcome.error_rate_change,
                'user_feedback': outcome.user_feedback
            })
        
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_confidence_report(self) -> Dict[str, Any]:
        """Get detailed confidence report."""
        report = {
            'confidence_score': self.metrics.confidence_score,
            'confidence_level': self._get_confidence_level(),
            'metrics': {
                'total_improvements': self.metrics.total_improvements,
                'success_rate': f"{self.metrics.success_rate:.1%}",
                'recent_success_rate': f"{self.metrics.recent_success_rate:.1%}",
                'days_active': self.metrics.days_since_start,
                'rollback_rate': f"{self.metrics.rollbacks_required / max(self.metrics.total_improvements, 1):.1%}"
            },
            'auto_apply_status': self._get_auto_apply_status(),
            'recommendations': self._get_recommendations()
        }
        
        return report
    
    def _get_confidence_level(self) -> str:
        """Get human-readable confidence level."""
        score = self.metrics.confidence_score
        
        if score >= 0.9:
            return "Very High"
        elif score >= 0.7:
            return "High"
        elif score >= 0.5:
            return "Medium"
        elif score >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def _get_auto_apply_status(self) -> Dict[str, Any]:
        """Get current auto-apply status."""
        max_risk = self._get_max_auto_risk()
        
        return {
            'enabled': self.metrics.days_since_start >= self.config['initial_manual_period_days'],
            'max_risk_level': max_risk,
            'eligible_types': [
                imp_type for imp_type, success_rate in self.metrics.success_rate_by_type.items()
                if success_rate >= self.config['min_success_rate_by_type']
            ]
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations for improving confidence."""
        recommendations = []
        
        if self.metrics.confidence_score < 0.5:
            recommendations.append("Continue manual review to build history")
        
        if self.metrics.success_rate < 0.95:
            recommendations.append("Focus on improving success rate before enabling auto-apply")
        
        if self.metrics.recent_success_rate < self.metrics.success_rate:
            recommendations.append("Recent performance declining - investigate recent failures")
        
        if self.metrics.rollbacks_required > 0:
            recommendations.append(f"Had {self.metrics.rollbacks_required} rollbacks - improve validation")
        
        # Type-specific recommendations
        for imp_type, success_rate in self.metrics.success_rate_by_type.items():
            if success_rate < 0.9:
                recommendations.append(f"Improve {imp_type} success rate (currently {success_rate:.1%})")
        
        return recommendations
    
    def assess_risk_level(self, improvement_type: ModificationType,
                         modification_details: Dict[str, Any]) -> RiskLevel:
        """Assess the risk level of an improvement."""
        risk_score = 0
        
        # Type-based risk
        type_risks = {
            ModificationType.OPTIMIZATION: 0.3,
            ModificationType.REFACTORING: 0.5,
            ModificationType.FEATURE_ADDITION: 0.7,
            ModificationType.BUG_FIX: 0.4,
            ModificationType.DOCUMENTATION: 0.1,
            ModificationType.TEST_ADDITION: 0.2,
            ModificationType.PERFORMANCE: 0.6,
            ModificationType.SECURITY: 0.8,
            ModificationType.EXTERNAL_INTEGRATION: 0.9
        }
        risk_score += type_risks.get(improvement_type, 0.5)
        
        # Size-based risk
        lines_changed = modification_details.get('lines_changed', 0)
        if lines_changed > 100:
            risk_score += 0.3
        elif lines_changed > 50:
            risk_score += 0.2
        elif lines_changed > 20:
            risk_score += 0.1
        
        # Complexity-based risk
        complexity_change = modification_details.get('complexity_change', 0)
        if complexity_change > 5:
            risk_score += 0.2
        elif complexity_change > 2:
            risk_score += 0.1
        
        # File criticality
        critical_files = ['__init__.py', 'main.py', 'core.py', 'api.py']
        target_file = modification_details.get('target_file', '')
        if any(critical in target_file for critical in critical_files):
            risk_score += 0.2
        
        # Normalize and map to risk level
        risk_score = min(1.0, risk_score)
        
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


def demonstrate_confidence():
    """Demonstrate the confidence system."""
    confidence = ProgressiveConfidence()
    
    print("=== Progressive Confidence System ===")
    print(f"Confidence Score: {confidence.metrics.confidence_score:.2f}")
    print(f"Days Active: {confidence.metrics.days_since_start}")
    print(f"Total Improvements: {confidence.metrics.total_improvements}")
    print(f"Success Rate: {confidence.metrics.success_rate:.1%}")
    
    # Test auto-apply decision
    should_apply, reason = confidence.should_auto_apply(
        ModificationType.OPTIMIZATION,
        RiskLevel.LOW
    )
    print(f"\nShould auto-apply optimization (low risk): {should_apply}")
    print(f"Reason: {reason}")
    
    # Get report
    report = confidence.get_confidence_report()
    print(f"\nConfidence Level: {report['confidence_level']}")
    print(f"Auto-apply enabled: {report['auto_apply_status']['enabled']}")
    print(f"Max risk level: {report['auto_apply_status']['max_risk_level']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")


if __name__ == "__main__":
    demonstrate_confidence()