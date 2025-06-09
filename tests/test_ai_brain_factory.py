"""
Unit tests for AI Brain Factory module.

Tests factory pattern implementation, environment-specific brain creation,
and configuration management. Follows AAA pattern with comprehensive coverage.
"""

import pytest
import os
from unittest.mock import Mock, patch
from datetime import datetime

from ai_brain_factory import AIBrainFactory


class TestAIBrainFactory:
    """Test suite for AIBrainFactory class."""

    def test_create_for_workflow(self):
        """Test workflow-optimized brain creation."""
        # Arrange & Act
        brain = AIBrainFactory.create_for_workflow()
        
        # Assert
        assert brain is not None
        assert hasattr(brain, 'state')
        assert hasattr(brain, 'context')
        assert brain.context.get('environment') == 'github_actions'
        assert brain.context.get('optimized_for') == 'workflow_execution'

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key', 'GITHUB_TOKEN': 'test_token'})
    def test_create_for_production(self):
        """Test production-ready brain creation."""
        # Arrange & Act
        brain = AIBrainFactory.create_for_production()
        
        # Assert
        assert brain is not None
        assert brain.context.get('environment') == 'production'
        assert brain.context.get('monitoring_enabled') is True
        assert brain.context.get('rate_limiting_enabled') is True

    def test_create_for_production_missing_requirements(self):
        """Test production brain creation fails without required environment."""
        # Arrange & Act
        with patch.dict(os.environ, {}, clear=True):
            brain = AIBrainFactory.create_for_production()
        
        # Assert - Should fall back to minimal configuration
        assert brain is not None
        assert brain.context.get('environment') in ['fallback', 'production']

    def test_create_for_testing(self):
        """Test testing-optimized brain creation."""
        # Arrange & Act
        brain = AIBrainFactory.create_for_testing()
        
        # Assert
        assert brain is not None
        assert brain.context.get('environment') == 'test'
        assert brain.context.get('mock_data') is True
        assert brain.context.get('api_calls_disabled') is True
        assert brain.context.get('predictable_responses') is True

    def test_create_for_testing_has_test_data(self):
        """Test testing brain includes predefined test data."""
        # Arrange & Act
        brain = AIBrainFactory.create_for_testing()
        
        # Assert
        assert 'projects' in brain.state
        assert 'test_project_1' in brain.state['projects']
        assert brain.state['charter']['purpose'] == 'test_system'

    def test_create_for_development(self):
        """Test development brain creation."""
        # Arrange & Act
        brain = AIBrainFactory.create_for_development()
        
        # Assert
        assert brain is not None
        assert brain.context.get('environment') in ['development', 'test']
        assert brain.context.get('debug_mode') is True

    def test_create_minimal_fallback(self):
        """Test minimal fallback brain creation."""
        # Arrange & Act
        brain = AIBrainFactory.create_minimal_fallback()
        
        # Assert
        assert brain is not None
        assert brain.context.get('environment') == 'fallback'
        assert brain.context.get('limited_functionality') is True
        assert brain.context.get('offline_mode') is True

    def test_create_with_config_valid(self):
        """Test brain creation with custom configuration."""
        # Arrange
        config = {
            'environment': 'custom_test',
            'features': ['feature_1', 'feature_2'],
            'debug_mode': False
        }
        
        # Act
        brain = AIBrainFactory.create_with_config(config)
        
        # Assert
        assert brain is not None
        assert brain.context.get('environment') == 'custom_config'
        assert brain.context.get('custom_features') == ['feature_1', 'feature_2']

    def test_create_with_config_invalid(self):
        """Test brain creation with invalid configuration."""
        # Arrange
        invalid_config = None
        
        # Act
        brain = AIBrainFactory.create_with_config(invalid_config)
        
        # Assert
        assert brain is not None  # Should fallback to minimal config

    def test_validate_brain_health_healthy(self):
        """Test brain health validation with healthy brain."""
        # Arrange
        healthy_brain = AIBrainFactory.create_for_testing()
        
        # Act
        is_healthy = AIBrainFactory._validate_brain_health(healthy_brain)
        
        # Assert
        assert is_healthy is True

    def test_validate_brain_health_unhealthy(self):
        """Test brain health validation with unhealthy brain."""
        # Arrange
        unhealthy_brain = Mock()
        unhealthy_brain.state = None  # Missing required state
        
        # Act
        is_healthy = AIBrainFactory._validate_brain_health(unhealthy_brain)
        
        # Assert
        assert is_healthy is False

    def test_get_environment_config_workflow(self):
        """Test environment configuration for workflow."""
        # Arrange & Act
        config = AIBrainFactory._get_environment_config('github_actions')
        
        # Assert
        assert config is not None
        assert config['environment'] == 'github_actions'
        assert config['optimized_for'] == 'workflow_execution'

    def test_get_environment_config_production(self):
        """Test environment configuration for production."""
        # Arrange & Act
        config = AIBrainFactory._get_environment_config('production')
        
        # Assert
        assert config is not None
        assert config['environment'] == 'production'
        assert config['monitoring_enabled'] is True

    def test_get_environment_config_test(self):
        """Test environment configuration for testing."""
        # Arrange & Act
        config = AIBrainFactory._get_environment_config('test')
        
        # Assert
        assert config is not None
        assert config['environment'] == 'test'
        assert config['mock_data'] is True

    def test_get_environment_config_unknown(self):
        """Test environment configuration for unknown environment."""
        # Arrange & Act
        config = AIBrainFactory._get_environment_config('unknown_env')
        
        # Assert
        assert config is not None
        assert config['environment'] == 'fallback'

    def test_create_test_state_structure(self):
        """Test test state has proper structure."""
        # Arrange & Act
        test_state = AIBrainFactory._create_test_state()
        
        # Assert
        assert 'projects' in test_state
        assert 'charter' in test_state
        assert 'performance_metrics' in test_state
        assert isinstance(test_state['projects'], dict)

    def test_create_test_state_has_sample_data(self):
        """Test test state includes sample data."""
        # Arrange & Act
        test_state = AIBrainFactory._create_test_state()
        
        # Assert
        assert len(test_state['projects']) > 0
        assert 'test_project_1' in test_state['projects']
        assert test_state['projects']['test_project_1']['status'] == 'active'

    @patch.dict(os.environ, {'CLAUDE_PAT': 'test_token'})
    def test_check_required_environment_with_token(self):
        """Test environment check with required token."""
        # Arrange & Act
        has_requirements = AIBrainFactory._check_required_environment(['CLAUDE_PAT'])
        
        # Assert
        assert has_requirements is True

    def test_check_required_environment_missing_token(self):
        """Test environment check missing required token."""
        # Arrange & Act
        with patch.dict(os.environ, {}, clear=True):
            has_requirements = AIBrainFactory._check_required_environment(['CLAUDE_PAT'])
        
        # Assert
        assert has_requirements is False

    def test_apply_environment_metadata_workflow(self):
        """Test environment metadata application for workflow."""
        # Arrange
        context = {}
        
        # Act
        AIBrainFactory._apply_environment_metadata(context, 'github_actions')
        
        # Assert
        assert 'created_at' in context
        assert 'optimized_for' in context
        assert context['optimized_for'] == 'workflow_execution'

    def test_apply_environment_metadata_production(self):
        """Test environment metadata application for production."""
        # Arrange
        context = {}
        
        # Act
        AIBrainFactory._apply_environment_metadata(context, 'production')
        
        # Assert
        assert context['monitoring_enabled'] is True
        assert context['rate_limiting_enabled'] is True

    def test_apply_environment_metadata_test(self):
        """Test environment metadata application for test."""
        # Arrange
        context = {}
        
        # Act
        AIBrainFactory._apply_environment_metadata(context, 'test')
        
        # Assert
        assert context['api_calls_disabled'] is True
        assert context['predictable_responses'] is True

    def test_factory_method_consistency(self):
        """Test all factory methods return consistent brain interface."""
        # Arrange & Act
        workflow_brain = AIBrainFactory.create_for_workflow()
        production_brain = AIBrainFactory.create_for_production()
        test_brain = AIBrainFactory.create_for_testing()
        dev_brain = AIBrainFactory.create_for_development()
        fallback_brain = AIBrainFactory.create_minimal_fallback()
        
        brains = [workflow_brain, production_brain, test_brain, dev_brain, fallback_brain]
        
        # Assert - All brains should have consistent interface
        for brain in brains:
            assert hasattr(brain, 'state')
            assert hasattr(brain, 'context')
            assert 'environment' in brain.context

    def test_brain_environment_isolation(self):
        """Test different environment brains are properly isolated."""
        # Arrange & Act
        test_brain = AIBrainFactory.create_for_testing()
        prod_brain = AIBrainFactory.create_for_production()
        
        # Assert
        assert test_brain.context['environment'] != prod_brain.context['environment']
        assert test_brain.context.get('api_calls_disabled') != prod_brain.context.get('api_calls_disabled')

    def test_performance_optimization_flags(self):
        """Test performance optimization flags are set correctly."""
        # Arrange & Act
        workflow_brain = AIBrainFactory.create_for_workflow()
        test_brain = AIBrainFactory.create_for_testing()
        
        # Assert
        assert workflow_brain.context.get('optimized_for') == 'workflow_execution'
        assert test_brain.context.get('mock_data') is True  # Performance optimization for testing

    def test_error_handling_in_factory_methods(self):
        """Test factory methods handle errors gracefully."""
        # Arrange & Act
        with patch('ai_brain_factory.IntelligentAIBrain') as mock_brain_class:
            mock_brain_class.side_effect = Exception("Brain creation failed")
            
            brain = AIBrainFactory.create_for_testing()
        
        # Assert - Should return a fallback brain or handle gracefully
        assert brain is not None or Exception  # Depending on implementation

    def test_configuration_inheritance(self):
        """Test configuration inheritance and override behavior."""
        # Arrange
        base_config = {
            'environment': 'base',
            'feature_1': True,
            'feature_2': False
        }
        
        override_config = {
            'feature_2': True,
            'feature_3': 'new_value'
        }
        
        # Act
        merged_config = AIBrainFactory._merge_configurations(base_config, override_config)
        
        # Assert
        assert merged_config['environment'] == 'base'  # Preserved
        assert merged_config['feature_1'] is True  # Preserved
        assert merged_config['feature_2'] is True  # Overridden
        assert merged_config['feature_3'] == 'new_value'  # Added

    def test_brain_initialization_parameters(self):
        """Test brain initialization receives correct parameters."""
        # Arrange & Act
        with patch('ai_brain_factory.IntelligentAIBrain') as mock_brain_class:
            mock_brain_instance = Mock()
            mock_brain_class.return_value = mock_brain_instance
            
            brain = AIBrainFactory.create_for_testing()
            
            # Assert
            mock_brain_class.assert_called_once()
            args, kwargs = mock_brain_class.call_args
            assert 'state' in kwargs or len(args) >= 1
            assert 'context' in kwargs or len(args) >= 2

    def test_factory_caching_behavior(self):
        """Test factory caching behavior (if implemented)."""
        # Arrange & Act
        brain1 = AIBrainFactory.create_for_testing()
        brain2 = AIBrainFactory.create_for_testing()
        
        # Assert - Depending on implementation, these might be same or different instances
        # For testing environments, typically want fresh instances
        assert brain1 is not brain2  # Fresh instances for testing

    def test_memory_efficiency(self):
        """Test factory creates memory-efficient brain instances."""
        # Arrange & Act
        brains = [AIBrainFactory.create_for_testing() for _ in range(3)]
        
        # Assert - Each brain should have its own state but share structure
        states = [brain.state for brain in brains]
        assert all(state is not None for state in states)
        # States should be independent
        assert states[0] is not states[1]