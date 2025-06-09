"""
Unit tests for AI Brain module.

Tests core intelligence engine, decision-making, adaptive algorithms, and meta-learning.
Follows AAA pattern with comprehensive coverage of AI brain functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from ai_brain import IntelligentAIBrain


class TestIntelligentAIBrain:
    """Test suite for IntelligentAIBrain class."""

    def test_init_default_state(self):
        """Test AI Brain initialization with default state."""
        # Arrange & Act
        brain = IntelligentAIBrain()
        
        # Assert
        assert brain.state is not None
        assert brain.context is not None
        assert brain.decision_history == []
        assert brain.performance_metrics == {}

    def test_init_with_custom_state(self):
        """Test AI Brain initialization with custom state."""
        # Arrange
        custom_state = {'test_key': 'test_value'}
        custom_context = {'environment': 'test'}
        
        # Act
        brain = IntelligentAIBrain(state=custom_state, context=custom_context)
        
        # Assert
        assert brain.state == custom_state
        assert brain.context == custom_context

    def test_action_types_structure(self):
        """Test ACTION_TYPES constant has required structure."""
        # Arrange & Act
        action_types = IntelligentAIBrain.ACTION_TYPES
        
        # Assert
        assert "GENERATE_TASKS" in action_types
        assert "REVIEW_TASKS" in action_types
        assert "PRIORITIZE_TASKS" in action_types
        assert "ANALYZE_PERFORMANCE" in action_types
        assert "UPDATE_DASHBOARD" in action_types
        
        # Check structure of each action type
        for action_name, action_data in action_types.items():
            assert "base_score" in action_data
            assert "goal_alignment" in action_data
            assert isinstance(action_data["base_score"], (int, float))
            assert isinstance(action_data["goal_alignment"], dict)

    def test_calculate_action_score_basic(self):
        """Test basic action score calculation."""
        # Arrange
        brain = IntelligentAIBrain()
        context = {
            'current_goals': ['innovation', 'efficiency'],
            'system_state': 'active',
            'resource_availability': 0.8
        }
        
        # Act
        score = brain.calculate_action_score("GENERATE_TASKS", context)
        
        # Assert
        assert isinstance(score, (int, float))
        assert score > 0

    def test_calculate_action_score_with_goal_alignment(self):
        """Test action score calculation with goal alignment."""
        # Arrange
        brain = IntelligentAIBrain()
        innovation_context = {
            'current_goals': ['innovation'],
            'system_state': 'active'
        }
        efficiency_context = {
            'current_goals': ['efficiency'],
            'system_state': 'active'
        }
        
        # Act
        innovation_score = brain.calculate_action_score("GENERATE_TASKS", innovation_context)
        efficiency_score = brain.calculate_action_score("PRIORITIZE_TASKS", efficiency_context)
        
        # Assert
        assert innovation_score > 0
        assert efficiency_score > 0
        # PRIORITIZE_TASKS has higher efficiency alignment than GENERATE_TASKS

    def test_decide_best_action_single_option(self):
        """Test decision making with single action option."""
        # Arrange
        brain = IntelligentAIBrain()
        context = {'current_goals': ['planning']}
        available_actions = ["GENERATE_TASKS"]
        
        # Act
        decision = brain.decide_best_action(context, available_actions)
        
        # Assert
        assert decision is not None
        assert decision['action'] == "GENERATE_TASKS"
        assert 'score' in decision
        assert 'reasoning' in decision

    def test_decide_best_action_multiple_options(self):
        """Test decision making with multiple action options."""
        # Arrange
        brain = IntelligentAIBrain()
        context = {'current_goals': ['efficiency', 'optimization']}
        available_actions = ["GENERATE_TASKS", "PRIORITIZE_TASKS", "ANALYZE_PERFORMANCE"]
        
        # Act
        decision = brain.decide_best_action(context, available_actions)
        
        # Assert
        assert decision is not None
        assert decision['action'] in available_actions
        assert decision['score'] > 0

    def test_decide_best_action_empty_options(self):
        """Test decision making with no available actions."""
        # Arrange
        brain = IntelligentAIBrain()
        context = {'current_goals': ['efficiency']}
        available_actions = []
        
        # Act
        decision = brain.decide_best_action(context, available_actions)
        
        # Assert
        assert decision is None

    def test_learn_from_outcome_success(self):
        """Test learning from successful outcomes."""
        # Arrange
        brain = IntelligentAIBrain()
        decision = {
            'action': 'GENERATE_TASKS',
            'score': 85,
            'context': {'goals': ['innovation']},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        outcome = {
            'success': True,
            'metrics': {'tasks_created': 5, 'quality_score': 9.2},
            'feedback': 'Excellent task generation'
        }
        
        # Act
        learning = brain.learn_from_outcome(decision, outcome)
        
        # Assert
        assert learning is not None
        assert learning['action'] == 'GENERATE_TASKS'
        assert learning['outcome_success'] is True
        assert 'learning_value' in learning

    def test_learn_from_outcome_failure(self):
        """Test learning from failed outcomes."""
        # Arrange
        brain = IntelligentAIBrain()
        decision = {
            'action': 'REVIEW_TASKS',
            'score': 70,
            'context': {'goals': ['quality']},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        outcome = {
            'success': False,
            'error': 'Task review failed due to timeout',
            'metrics': {'completion_rate': 0.2}
        }
        
        # Act
        learning = brain.learn_from_outcome(decision, outcome)
        
        # Assert
        assert learning is not None
        assert learning['action'] == 'REVIEW_TASKS'
        assert learning['outcome_success'] is False
        assert 'error_analysis' in learning

    def test_update_performance_metrics(self):
        """Test performance metrics updating."""
        # Arrange
        brain = IntelligentAIBrain()
        initial_metrics = brain.performance_metrics.copy()
        
        learning_data = {
            'action': 'GENERATE_TASKS',
            'outcome_success': True,
            'learning_value': 0.85,
            'execution_time': 2.5
        }
        
        # Act
        brain.update_performance_metrics(learning_data)
        
        # Assert
        assert brain.performance_metrics != initial_metrics
        assert 'GENERATE_TASKS' in brain.performance_metrics

    def test_get_performance_summary(self):
        """Test performance summary generation."""
        # Arrange
        brain = IntelligentAIBrain()
        brain.performance_metrics = {
            'GENERATE_TASKS': {
                'total_executions': 10,
                'success_count': 8,
                'average_score': 82.5,
                'learning_trend': 0.15
            }
        }
        brain.decision_history = [
            {'action': 'GENERATE_TASKS', 'timestamp': datetime.now().isoformat()},
            {'action': 'REVIEW_TASKS', 'timestamp': datetime.now().isoformat()}
        ]
        
        # Act
        summary = brain.get_performance_summary()
        
        # Assert
        assert 'total_decisions' in summary
        assert 'performance_by_action' in summary
        assert 'overall_success_rate' in summary
        assert summary['total_decisions'] == 2

    def test_adaptive_scoring_with_history(self):
        """Test adaptive scoring based on historical performance."""
        # Arrange
        brain = IntelligentAIBrain()
        brain.performance_metrics = {
            'GENERATE_TASKS': {
                'success_rate': 0.9,
                'average_score': 88.0,
                'learning_trend': 0.2
            }
        }
        context = {'current_goals': ['innovation']}
        
        # Act
        score_without_history = IntelligentAIBrain().calculate_action_score("GENERATE_TASKS", context)
        score_with_history = brain.calculate_action_score("GENERATE_TASKS", context)
        
        # Assert
        assert score_with_history != score_without_history
        # With good historical performance, score should be adjusted

    def test_context_aware_decision_making(self):
        """Test decision making adapts to different contexts."""
        # Arrange
        brain = IntelligentAIBrain()
        
        planning_context = {
            'current_goals': ['planning', 'organization'],
            'system_load': 'low',
            'time_of_day': 'morning'
        }
        
        crisis_context = {
            'current_goals': ['reliability', 'stability'],
            'system_load': 'high',
            'critical_issues': 3
        }
        
        actions = ["GENERATE_TASKS", "ANALYZE_PERFORMANCE", "REVIEW_TASKS"]
        
        # Act
        planning_decision = brain.decide_best_action(planning_context, actions)
        crisis_decision = brain.decide_best_action(crisis_context, actions)
        
        # Assert
        assert planning_decision['action'] != crisis_decision['action'] or \
               planning_decision['reasoning'] != crisis_decision['reasoning']

    def test_goal_priority_weighting(self):
        """Test goal priority affects action scoring."""
        # Arrange
        brain = IntelligentAIBrain()
        
        high_innovation_context = {
            'current_goals': ['innovation'],
            'goal_weights': {'innovation': 1.0, 'efficiency': 0.3}
        }
        
        high_efficiency_context = {
            'current_goals': ['efficiency'],
            'goal_weights': {'innovation': 0.3, 'efficiency': 1.0}
        }
        
        # Act
        innovation_score = brain.calculate_action_score("GENERATE_TASKS", high_innovation_context)
        efficiency_score = brain.calculate_action_score("PRIORITIZE_TASKS", high_efficiency_context)
        
        # Assert
        assert innovation_score > 0
        assert efficiency_score > 0

    def test_decision_history_tracking(self):
        """Test decision history is properly tracked."""
        # Arrange
        brain = IntelligentAIBrain()
        context = {'current_goals': ['testing']}
        actions = ["GENERATE_TASKS"]
        
        initial_history_length = len(brain.decision_history)
        
        # Act
        decision = brain.decide_best_action(context, actions)
        
        # Assert
        assert len(brain.decision_history) == initial_history_length + 1
        assert brain.decision_history[-1]['action'] == decision['action']

    def test_meta_learning_adaptation(self):
        """Test meta-learning capabilities."""
        # Arrange
        brain = IntelligentAIBrain()
        
        # Simulate multiple learning cycles
        for i in range(5):
            decision = {
                'action': 'GENERATE_TASKS',
                'score': 80 + i,
                'context': {'goals': ['innovation']},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            outcome = {
                'success': True,
                'metrics': {'quality_score': 8.0 + i * 0.2}
            }
            brain.learn_from_outcome(decision, outcome)
        
        # Act
        meta_learning_insights = brain.get_meta_learning_insights()
        
        # Assert
        assert meta_learning_insights is not None
        assert 'improvement_patterns' in meta_learning_insights

    @patch('ai_brain.HTTPAIClient')
    def test_ai_client_integration(self, mock_ai_client):
        """Test integration with AI client for enhanced reasoning."""
        # Arrange
        mock_client = Mock()
        mock_client.generate_enhanced_response.return_value = {
            'content': 'AI-generated reasoning',
            'confidence': 0.85
        }
        mock_ai_client.return_value = mock_client
        
        brain = IntelligentAIBrain()
        brain.ai_client = mock_client
        
        context = {'current_goals': ['innovation']}
        
        # Act
        reasoning = brain.generate_reasoning(context, "GENERATE_TASKS")
        
        # Assert
        assert reasoning is not None
        assert 'content' in reasoning

    def test_error_handling_invalid_action(self):
        """Test error handling for invalid actions."""
        # Arrange
        brain = IntelligentAIBrain()
        context = {'current_goals': ['testing']}
        
        # Act
        score = brain.calculate_action_score("INVALID_ACTION", context)
        
        # Assert
        assert score == 0  # Should return 0 for invalid actions

    def test_state_persistence_and_recovery(self):
        """Test state can be persisted and recovered."""
        # Arrange
        brain = IntelligentAIBrain()
        brain.state = {'test_data': 'important_value'}
        brain.performance_metrics = {'test_metric': 42}
        
        # Act
        state_snapshot = brain.get_state_snapshot()
        new_brain = IntelligentAIBrain()
        new_brain.restore_from_snapshot(state_snapshot)
        
        # Assert
        assert new_brain.state['test_data'] == 'important_value'
        assert new_brain.performance_metrics['test_metric'] == 42

    def test_concurrent_decision_making(self):
        """Test brain handles concurrent decision requests."""
        # Arrange
        brain = IntelligentAIBrain()
        context = {'current_goals': ['efficiency']}
        actions = ["GENERATE_TASKS", "PRIORITIZE_TASKS"]
        
        # Act
        decision1 = brain.decide_best_action(context, actions)
        decision2 = brain.decide_best_action(context, actions)
        
        # Assert
        assert decision1 is not None
        assert decision2 is not None
        # Decisions should be consistent for same context
        assert decision1['action'] == decision2['action']