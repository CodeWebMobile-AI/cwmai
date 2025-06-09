#!/usr/bin/env python3
"""
Comprehensive unit tests for AIBrainFactory module.

Tests cover:
- Factory method patterns for different environments
- Environment-specific brain configurations
- Error handling and fallback mechanisms
- State loading with repository discovery
- External dependency mocking
"""

import unittest
import os
import logging
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.ai_brain_factory import AIBrainFactory


class TestAIBrainFactoryWorkflowCreation(unittest.TestCase):
    """Test AIBrainFactory workflow creation functionality."""
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    @patch.dict(os.environ, {'GITHUB_ACTIONS': 'true'})
    def test_create_for_workflow_in_github_actions(self, mock_brain_class, mock_state_manager_class):
        """Test creating AI brain for workflow in GitHub Actions environment."""
        # Arrange
        mock_state_manager = Mock()
        mock_state = {'projects': {'proj1': {}, 'proj2': {}}}
        mock_state_manager.load_state_with_repository_discovery.return_value = mock_state
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_for_workflow()
        
        # Assert
        mock_state_manager_class.assert_called_once()
        mock_state_manager.load_state_with_repository_discovery.assert_called_once()
        mock_brain_class.assert_called_once()
        self.assertEqual(result, mock_brain)
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_workflow_without_github_actions(self, mock_brain_class, mock_state_manager_class):
        """Test creating AI brain for workflow outside GitHub Actions environment."""
        # Arrange
        mock_state_manager = Mock()
        mock_state = {'projects': {}}
        mock_state_manager.load_state_with_repository_discovery.return_value = mock_state
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain_class.return_value = mock_brain
        
        # Act
        with patch.dict(os.environ, {}, clear=True):  # No GITHUB_ACTIONS
            result = AIBrainFactory.create_for_workflow()
        
        # Assert
        mock_state_manager_class.assert_called_once()
        self.assertEqual(result, mock_brain)
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_workflow_with_repository_discovery_failure(self, mock_brain_class, mock_state_manager_class):
        """Test workflow creation when repository discovery fails."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state_with_repository_discovery.side_effect = Exception("Discovery failed")
        mock_state_manager.load_workflow_state.return_value = {'projects': {}}
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_for_workflow()
        
        # Assert
        mock_state_manager.load_state_with_repository_discovery.assert_called_once()
        mock_state_manager.load_workflow_state.assert_called_once()
        self.assertEqual(result, mock_brain)
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_workflow_with_complete_fallback(self, mock_brain_class, mock_state_manager_class):
        """Test workflow creation when all state loading methods fail."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state_with_repository_discovery.side_effect = Exception("Discovery failed")
        mock_state_manager.load_workflow_state.side_effect = AttributeError("Method not found")
        mock_state_manager.load_state.return_value = {'projects': {}}
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_for_workflow()
        
        # Assert
        mock_state_manager.load_state_with_repository_discovery.assert_called_once()
        mock_state_manager.load_workflow_state.assert_called_once()
        mock_state_manager.load_state.assert_called_once()
        self.assertEqual(result, mock_brain)


class TestAIBrainFactoryProductionCreation(unittest.TestCase):
    """Test AIBrainFactory production creation functionality."""
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_production_success(self, mock_brain_class, mock_state_manager_class):
        """Test creating AI brain for production environment."""
        # Arrange
        mock_state_manager = Mock()
        mock_state = {'projects': {'prod_proj': {}}}
        mock_state_manager.load_state.return_value = mock_state
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_for_production()
        
        # Assert
        mock_state_manager_class.assert_called_once()
        mock_state_manager.load_state.assert_called_once()
        mock_brain_class.assert_called_once()
        
        # Check that production-specific context is set
        self.assertEqual(mock_brain.context['environment'], 'production')
        self.assertEqual(mock_brain.context['monitoring_enabled'], True)
        self.assertEqual(mock_brain.context['optimized_for'], 'reliability')
        self.assertIn('created_at', mock_brain.context)
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_production_validates_health(self, mock_brain_class, mock_state_manager_class):
        """Test that production creation validates brain health."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = {'projects': {}}
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain_class.return_value = mock_brain
        
        # Mock health validation to return False (unhealthy)
        with patch.object(AIBrainFactory, '_validate_brain_health', return_value=False):
            # Act & Assert
            with self.assertRaises(RuntimeError) as context:
                AIBrainFactory.create_for_production()
            
            self.assertIn("Failed health validation", str(context.exception))


class TestAIBrainFactoryTestingCreation(unittest.TestCase):
    """Test AIBrainFactory testing creation functionality."""
    
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_testing_success(self, mock_brain_class):
        """Test creating AI brain for testing environment."""
        # Arrange
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain.state = {}
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_for_testing()
        
        # Assert
        mock_brain_class.assert_called_once()
        
        # Check testing-specific context
        self.assertEqual(mock_brain.context['environment'], 'test')
        self.assertEqual(mock_brain.context['mock_data'], True)
        self.assertEqual(mock_brain.context['predictable_responses'], True)
        self.assertEqual(mock_brain.context['api_calls_disabled'], True)
        
        # Check that test data is populated
        self.assertIn('test_project_1', mock_brain.state['projects'])
        self.assertEqual(mock_brain.state['charter']['purpose'], 'test_system')
    
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_testing_provides_deterministic_data(self, mock_brain_class):
        """Test that testing brain provides deterministic test data."""
        # Arrange
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain.state = {}
        mock_brain_class.return_value = mock_brain
        
        # Act
        result1 = AIBrainFactory.create_for_testing()
        result2 = AIBrainFactory.create_for_testing()
        
        # Assert - Both instances should have identical test data
        self.assertEqual(result1.state['projects'], result2.state['projects'])
        self.assertEqual(result1.state['charter'], result2.state['charter'])


class TestAIBrainFactoryDevelopmentCreation(unittest.TestCase):
    """Test AIBrainFactory development creation functionality."""
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_development_success(self, mock_brain_class, mock_state_manager_class):
        """Test creating AI brain for development environment."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = {'projects': {}}
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_for_development()
        
        # Assert
        mock_brain_class.assert_called_once()
        self.assertEqual(mock_brain.context['environment'], 'development')
        self.assertEqual(mock_brain.context['debug_mode'], True)
        self.assertEqual(mock_brain.context['enhanced_logging'], True)
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_development_with_state_loading_failure(self, mock_brain_class, mock_state_manager_class):
        """Test development creation when state loading fails."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state.side_effect = Exception("State loading failed")
        mock_state_manager_class.return_value = mock_state_manager
        
        # Act
        result = AIBrainFactory.create_for_development()
        
        # Assert - Should fallback to test environment
        self.assertEqual(result.context['environment'], 'test')


class TestAIBrainFactoryFallbackCreation(unittest.TestCase):
    """Test AIBrainFactory fallback creation functionality."""
    
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_minimal_fallback(self, mock_brain_class):
        """Test creating minimal fallback AI brain."""
        # Arrange
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain.state = {}
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_minimal_fallback()
        
        # Assert
        mock_brain_class.assert_called_once()
        self.assertEqual(mock_brain.context['environment'], 'fallback')
        self.assertEqual(mock_brain.context['limited_functionality'], True)
        self.assertEqual(mock_brain.context['safe_mode'], True)
        self.assertIn('minimal_state', mock_brain.state)


class TestAIBrainFactoryCustomConfig(unittest.TestCase):
    """Test AIBrainFactory custom configuration functionality."""
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_with_config_success(self, mock_brain_class, mock_state_manager_class):
        """Test creating AI brain with custom configuration."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state.return_value = {'projects': {}}
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain_class.return_value = mock_brain
        
        custom_config = {
            'environment': 'staging',
            'features': ['feature1', 'feature2'],
            'debug_level': 'verbose'
        }
        
        # Act
        result = AIBrainFactory.create_with_config(custom_config)
        
        # Assert
        mock_brain_class.assert_called_once()
        self.assertEqual(mock_brain.context['environment'], 'custom_config')
        self.assertEqual(mock_brain.context['user_config'], custom_config)
        self.assertIn('created_at', mock_brain.context)
    
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_with_config_empty_config(self, mock_brain_class):
        """Test creating AI brain with empty configuration."""
        # Arrange
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain_class.return_value = mock_brain
        
        # Act
        result = AIBrainFactory.create_with_config({})
        
        # Assert
        self.assertEqual(mock_brain.context['user_config'], {})


class TestAIBrainFactoryHealthValidation(unittest.TestCase):
    """Test AIBrainFactory health validation functionality."""
    
    def test_validate_brain_health_with_healthy_brain(self):
        """Test health validation with a healthy brain."""
        # Arrange
        healthy_brain = Mock()
        healthy_brain.state = {'projects': {}}
        healthy_brain.context = {'environment': 'test'}
        
        # Act
        is_healthy = AIBrainFactory._validate_brain_health(healthy_brain)
        
        # Assert
        self.assertTrue(is_healthy)
    
    def test_validate_brain_health_with_missing_state(self):
        """Test health validation with brain missing state."""
        # Arrange
        unhealthy_brain = Mock()
        unhealthy_brain.state = None
        unhealthy_brain.context = {'environment': 'test'}
        
        # Act
        is_healthy = AIBrainFactory._validate_brain_health(unhealthy_brain)
        
        # Assert
        self.assertFalse(is_healthy)
    
    def test_validate_brain_health_with_missing_context(self):
        """Test health validation with brain missing context."""
        # Arrange
        unhealthy_brain = Mock()
        unhealthy_brain.state = {'projects': {}}
        unhealthy_brain.context = None
        
        # Act
        is_healthy = AIBrainFactory._validate_brain_health(unhealthy_brain)
        
        # Assert
        self.assertFalse(is_healthy)
    
    def test_validate_brain_health_with_none_brain(self):
        """Test health validation with None brain."""
        # Act
        is_healthy = AIBrainFactory._validate_brain_health(None)
        
        # Assert
        self.assertFalse(is_healthy)


class TestAIBrainFactoryErrorHandling(unittest.TestCase):
    """Test AIBrainFactory error handling."""
    
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_workflow_handles_brain_creation_failure(self, mock_brain_class):
        """Test workflow creation when brain creation fails."""
        # Arrange
        mock_brain_class.side_effect = Exception("Brain creation failed")
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            AIBrainFactory.create_for_workflow()
        
        self.assertIn("Brain creation failed", str(context.exception))
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_production_handles_state_manager_failure(self, mock_brain_class, mock_state_manager_class):
        """Test production creation when StateManager fails."""
        # Arrange
        mock_state_manager_class.side_effect = Exception("StateManager initialization failed")
        
        # Act & Assert
        with self.assertRaises(Exception) as context:
            AIBrainFactory.create_for_production()
        
        self.assertIn("StateManager initialization failed", str(context.exception))
    
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_create_for_testing_handles_test_data_setup_failure(self, mock_brain_class):
        """Test testing creation when test data setup fails."""
        # Arrange
        mock_brain = Mock()
        mock_brain.context = {}
        mock_brain.state = None  # This should cause an error when trying to set test data
        mock_brain_class.return_value = mock_brain
        
        # Act - Should handle the error gracefully
        result = AIBrainFactory.create_for_testing()
        
        # Assert - Should still return a brain even if test data setup fails
        self.assertIsNotNone(result)


class TestAIBrainFactoryLogging(unittest.TestCase):
    """Test AIBrainFactory logging functionality."""
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_workflow_creation_logs_appropriately(self, mock_brain_class, mock_state_manager_class):
        """Test that workflow creation logs appropriate messages."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state_with_repository_discovery.return_value = {'projects': {'proj1': {}}}
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain_class.return_value = mock_brain
        
        # Act
        with patch('scripts.ai_brain_factory.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            AIBrainFactory.create_for_workflow()
            
            # Assert - Check that info logging was called
            mock_logger.info.assert_called()
    
    @patch('scripts.ai_brain_factory.StateManager')
    @patch('scripts.ai_brain_factory.IntelligentAIBrain')
    def test_workflow_creation_logs_warnings_for_failures(self, mock_brain_class, mock_state_manager_class):
        """Test that workflow creation logs warnings for failures."""
        # Arrange
        mock_state_manager = Mock()
        mock_state_manager.load_state_with_repository_discovery.side_effect = Exception("Discovery failed")
        mock_state_manager.load_workflow_state.return_value = {'projects': {}}
        mock_state_manager_class.return_value = mock_state_manager
        
        mock_brain = Mock()
        mock_brain_class.return_value = mock_brain
        
        # Act
        with patch('scripts.ai_brain_factory.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            AIBrainFactory.create_for_workflow()
            
            # Assert - Check that warning logging was called
            mock_logger.warning.assert_called()


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)