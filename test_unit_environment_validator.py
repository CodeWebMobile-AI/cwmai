#!/usr/bin/env python3
"""
Comprehensive unit tests for EnvironmentValidator module.

Tests cover:
- Environment variable validation
- API key validation
- Secret configuration management
- Validation status reporting
- Error handling and edge cases
- Security considerations for key validation
"""

import unittest
import os
from unittest.mock import Mock, patch
from dataclasses import dataclass
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

from scripts.environment_validator import EnvironmentValidator, ValidationStatus, SecretConfig


class TestValidationStatus(unittest.TestCase):
    """Test ValidationStatus enumeration."""
    
    def test_validation_status_values(self):
        """Test ValidationStatus enum has correct values."""
        # Assert
        self.assertEqual(ValidationStatus.AVAILABLE.value, "✅")
        self.assertEqual(ValidationStatus.MISSING.value, "❌")
        self.assertEqual(ValidationStatus.WARNING.value, "⚠️")
        self.assertEqual(ValidationStatus.INFO.value, "ℹ️")


class TestSecretConfig(unittest.TestCase):
    """Test SecretConfig dataclass."""
    
    def test_secret_config_creation_minimal(self):
        """Test SecretConfig creation with minimal parameters."""
        # Arrange & Act
        config = SecretConfig(
            name="TEST_SECRET",
            description="A test secret"
        )
        
        # Assert
        self.assertEqual(config.name, "TEST_SECRET")
        self.assertEqual(config.description, "A test secret")
        self.assertTrue(config.required)  # Default value
        self.assertIsNone(config.validation_func)  # Default value
        self.assertIsNone(config.documentation_url)  # Default value
    
    def test_secret_config_creation_full(self):
        """Test SecretConfig creation with all parameters."""
        # Arrange
        validation_func = lambda x: True
        
        # Act
        config = SecretConfig(
            name="FULL_SECRET",
            description="A full secret configuration",
            required=False,
            validation_func=validation_func,
            documentation_url="https://example.com/docs"
        )
        
        # Assert
        self.assertEqual(config.name, "FULL_SECRET")
        self.assertEqual(config.description, "A full secret configuration")
        self.assertFalse(config.required)
        self.assertEqual(config.validation_func, validation_func)
        self.assertEqual(config.documentation_url, "https://example.com/docs")


class TestEnvironmentValidatorInitialization(unittest.TestCase):
    """Test EnvironmentValidator initialization."""
    
    def test_init_creates_secrets_config(self):
        """Test that initialization creates secrets configuration."""
        # Arrange & Act
        validator = EnvironmentValidator()
        
        # Assert
        self.assertIsInstance(validator.secrets_config, list)
        self.assertGreater(len(validator.secrets_config), 0)
        
        # Check for required secrets
        secret_names = [config.name for config in validator.secrets_config]
        self.assertIn("ANTHROPIC_API_KEY", secret_names)
        self.assertIn("CLAUDE_PAT", secret_names)
    
    def test_init_anthropic_api_key_config(self):
        """Test Anthropic API key configuration."""
        # Arrange & Act
        validator = EnvironmentValidator()
        
        # Assert
        anthropic_config = next(
            config for config in validator.secrets_config 
            if config.name == "ANTHROPIC_API_KEY"
        )
        self.assertEqual(anthropic_config.description, "Anthropic API key for Claude AI access")
        self.assertTrue(anthropic_config.required)
        self.assertIsNotNone(anthropic_config.validation_func)
        self.assertEqual(anthropic_config.documentation_url, "https://console.anthropic.com/")
    
    def test_init_claude_pat_config(self):
        """Test Claude PAT configuration."""
        # Arrange & Act
        validator = EnvironmentValidator()
        
        # Assert
        claude_pat_config = next(
            config for config in validator.secrets_config 
            if config.name == "CLAUDE_PAT"
        )
        self.assertEqual(claude_pat_config.description, "GitHub Personal Access Token with repo permissions")
        self.assertTrue(claude_pat_config.required)
        self.assertIsNotNone(claude_pat_config.validation_func)
        self.assertEqual(claude_pat_config.documentation_url, "https://github.com/settings/tokens")


class TestEnvironmentValidatorAPIKeyValidation(unittest.TestCase):
    """Test API key validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = EnvironmentValidator()
    
    def test_validate_anthropic_key_valid_format(self):
        """Test Anthropic key validation with valid format."""
        # Arrange
        valid_key = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz"
        
        # Act
        is_valid = self.validator._validate_anthropic_key(valid_key)
        
        # Assert
        self.assertTrue(is_valid)
    
    def test_validate_anthropic_key_invalid_format(self):
        """Test Anthropic key validation with invalid format."""
        # Arrange
        invalid_keys = [
            "invalid_key",
            "sk-wrong-format",
            "",
            None,
            "sk-ant-api03-short"
        ]
        
        # Act & Assert
        for invalid_key in invalid_keys:
            with self.subTest(key=invalid_key):
                is_valid = self.validator._validate_anthropic_key(invalid_key)
                self.assertFalse(is_valid)
    
    def test_validate_github_token_valid_format(self):
        """Test GitHub token validation with valid formats."""
        # Arrange
        valid_tokens = [
            "ghp_abcdefghijklmnopqrstuvwxyz1234567890",  # Personal access token
            "gho_abcdefghijklmnopqrstuvwxyz1234567890",  # OAuth token
            "ghu_abcdefghijklmnopqrstuvwxyz1234567890",  # User token
            "ghs_abcdefghijklmnopqrstuvwxyz1234567890",  # Server token
            "ghr_abcdefghijklmnopqrstuvwxyz1234567890"   # Refresh token
        ]
        
        # Act & Assert
        for valid_token in valid_tokens:
            with self.subTest(token=valid_token):
                is_valid = self.validator._validate_github_token(valid_token)
                self.assertTrue(is_valid)
    
    def test_validate_github_token_invalid_format(self):
        """Test GitHub token validation with invalid formats."""
        # Arrange
        invalid_tokens = [
            "invalid_token",
            "ghp_short",
            "",
            None,
            "wrong_prefix_abcdefghijklmnopqrstuvwxyz1234567890"
        ]
        
        # Act & Assert
        for invalid_token in invalid_tokens:
            with self.subTest(token=invalid_token):
                is_valid = self.validator._validate_github_token(invalid_token)
                self.assertFalse(is_valid)
    
    def test_validate_openai_key_valid_format(self):
        """Test OpenAI key validation with valid format."""
        # Arrange
        valid_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz"
        
        # Act
        is_valid = self.validator._validate_openai_key(valid_key)
        
        # Assert
        self.assertTrue(is_valid)
    
    def test_validate_openai_key_invalid_format(self):
        """Test OpenAI key validation with invalid format."""
        # Arrange
        invalid_keys = [
            "invalid_key",
            "sk-short",
            "",
            None,
            "wrong-prefix-abcdefghijklmnopqrstuvwxyz1234567890"
        ]
        
        # Act & Assert
        for invalid_key in invalid_keys:
            with self.subTest(key=invalid_key):
                is_valid = self.validator._validate_openai_key(invalid_key)
                self.assertFalse(is_valid)


class TestEnvironmentValidatorValidateEnvironment(unittest.TestCase):
    """Test environment validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = EnvironmentValidator()
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-api03-valid_anthropic_key_1234567890abcdefghijklmnopqrstuvwxyz',
        'CLAUDE_PAT': 'ghp_valid_github_token_1234567890abcdefghijkl'
    })
    def test_validate_environment_all_required_present(self):
        """Test environment validation when all required secrets are present."""
        # Act
        results = self.validator.validate_environment()
        
        # Assert
        self.assertIsInstance(results, dict)
        
        # Check that required secrets are validated
        self.assertIn('ANTHROPIC_API_KEY', results)
        self.assertIn('CLAUDE_PAT', results)
        
        # Check status for required secrets
        self.assertEqual(results['ANTHROPIC_API_KEY']['status'], ValidationStatus.AVAILABLE)
        self.assertEqual(results['CLAUDE_PAT']['status'], ValidationStatus.AVAILABLE)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_missing_required_secrets(self):
        """Test environment validation when required secrets are missing."""
        # Act
        results = self.validator.validate_environment()
        
        # Assert
        # Check that missing required secrets are reported as missing
        self.assertEqual(results['ANTHROPIC_API_KEY']['status'], ValidationStatus.MISSING)
        self.assertEqual(results['CLAUDE_PAT']['status'], ValidationStatus.MISSING)
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'invalid_key',
        'CLAUDE_PAT': 'invalid_token'
    })
    def test_validate_environment_invalid_secrets(self):
        """Test environment validation with invalid secret formats."""
        # Act
        results = self.validator.validate_environment()
        
        # Assert
        # Check that invalid secrets are reported as warnings or missing
        self.assertIn(results['ANTHROPIC_API_KEY']['status'], [ValidationStatus.WARNING, ValidationStatus.MISSING])
        self.assertIn(results['CLAUDE_PAT']['status'], [ValidationStatus.WARNING, ValidationStatus.MISSING])
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-api03-valid_anthropic_key_1234567890abcdefghijklmnopqrstuvwxyz',
        'CLAUDE_PAT': 'ghp_valid_github_token_1234567890abcdefghijkl',
        'OPENAI_API_KEY': 'sk-valid_openai_key_1234567890abcdefghijklmnopqrstuvwxyz',
        'GEMINI_API_KEY': 'valid_gemini_key_1234567890'
    })
    def test_validate_environment_optional_secrets_present(self):
        """Test environment validation with optional secrets present."""
        # Act
        results = self.validator.validate_environment()
        
        # Assert
        # Check that optional secrets are included when present
        self.assertIn('OPENAI_API_KEY', results)
        self.assertIn('GEMINI_API_KEY', results)
        
        # Optional secrets should be validated too
        if 'OPENAI_API_KEY' in results:
            self.assertEqual(results['OPENAI_API_KEY']['status'], ValidationStatus.AVAILABLE)


class TestEnvironmentValidatorReporting(unittest.TestCase):
    """Test environment validation reporting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = EnvironmentValidator()
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-api03-valid_anthropic_key_1234567890abcdefghijklmnopqrstuvwxyz',
        'CLAUDE_PAT': 'ghp_valid_github_token_1234567890abcdefghijkl'
    })
    def test_print_validation_report_success(self):
        """Test printing validation report for successful validation."""
        # Arrange
        results = self.validator.validate_environment()
        
        # Act & Assert - Should not raise any exceptions
        try:
            self.validator.print_validation_report(results)
        except Exception as e:
            self.fail(f"print_validation_report raised an exception: {e}")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_print_validation_report_with_missing_secrets(self):
        """Test printing validation report with missing secrets."""
        # Arrange
        results = self.validator.validate_environment()
        
        # Act & Assert - Should not raise any exceptions
        try:
            self.validator.print_validation_report(results)
        except Exception as e:
            self.fail(f"print_validation_report raised an exception: {e}")
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-api03-valid_anthropic_key_1234567890abcdefghijklmnopqrstuvwxyz',
        'CLAUDE_PAT': 'ghp_valid_github_token_1234567890abcdefghijkl'
    })
    def test_is_environment_ready_with_valid_environment(self):
        """Test environment readiness check with valid environment."""
        # Arrange
        results = self.validator.validate_environment()
        
        # Act
        is_ready = self.validator.is_environment_ready(results)
        
        # Assert
        self.assertTrue(is_ready)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_is_environment_ready_with_missing_required_secrets(self):
        """Test environment readiness check with missing required secrets."""
        # Arrange
        results = self.validator.validate_environment()
        
        # Act
        is_ready = self.validator.is_environment_ready(results)
        
        # Assert
        self.assertFalse(is_ready)
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-api03-valid_anthropic_key_1234567890abcdefghijklmnopqrstuvwxyz'
        # Missing CLAUDE_PAT
    })
    def test_is_environment_ready_with_partial_required_secrets(self):
        """Test environment readiness check with partial required secrets."""
        # Arrange
        results = self.validator.validate_environment()
        
        # Act
        is_ready = self.validator.is_environment_ready(results)
        
        # Assert
        self.assertFalse(is_ready)


class TestEnvironmentValidatorSecurityConsiderations(unittest.TestCase):
    """Test security aspects of environment validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = EnvironmentValidator()
    
    def test_validation_does_not_log_secrets(self):
        """Test that validation methods don't expose secret values."""
        # Arrange
        secret_value = "sk-ant-api03-secret_key_that_should_not_be_logged_1234567890"
        
        # Act
        with patch('builtins.print') as mock_print:
            self.validator._validate_anthropic_key(secret_value)
            
            # Assert - Check that the secret value is not in any print calls
            for call in mock_print.call_args_list:
                call_str = str(call)
                self.assertNotIn(secret_value, call_str)
    
    def test_validation_result_does_not_expose_secrets(self):
        """Test that validation results don't expose secret values."""
        # Arrange
        secret_key = "sk-ant-api03-secret_that_should_not_be_exposed_1234567890"
        
        # Act
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': secret_key}):
            results = self.validator.validate_environment()
        
        # Assert
        # Check that the secret is not exposed in the results
        results_str = str(results)
        self.assertNotIn(secret_key, results_str)
        
        # But check that the key name is present
        self.assertIn('ANTHROPIC_API_KEY', results)


class TestEnvironmentValidatorErrorHandling(unittest.TestCase):
    """Test error handling in environment validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = EnvironmentValidator()
    
    def test_validate_anthropic_key_with_none(self):
        """Test Anthropic key validation with None value."""
        # Act
        is_valid = self.validator._validate_anthropic_key(None)
        
        # Assert
        self.assertFalse(is_valid)
    
    def test_validate_github_token_with_none(self):
        """Test GitHub token validation with None value."""
        # Act
        is_valid = self.validator._validate_github_token(None)
        
        # Assert
        self.assertFalse(is_valid)
    
    def test_validate_environment_handles_validation_function_exception(self):
        """Test that environment validation handles validation function exceptions."""
        # Arrange
        # Create a validator with a faulty validation function
        validator = EnvironmentValidator()
        
        # Mock a validation function that raises an exception
        def faulty_validation(value):
            raise Exception("Validation function error")
        
        # Replace the validation function for one of the secrets
        for config in validator.secrets_config:
            if config.name == "ANTHROPIC_API_KEY":
                config.validation_func = faulty_validation
                break
        
        # Act
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'some_value'}):
            results = validator.validate_environment()
        
        # Assert
        # Should handle the exception gracefully
        self.assertIn('ANTHROPIC_API_KEY', results)
        # The status should indicate an issue (not AVAILABLE)
        self.assertNotEqual(results['ANTHROPIC_API_KEY']['status'], ValidationStatus.AVAILABLE)
    
    def test_print_validation_report_handles_malformed_results(self):
        """Test that print_validation_report handles malformed results gracefully."""
        # Arrange
        malformed_results = {
            'INVALID_KEY': "not a dict",
            'ANOTHER_KEY': {'missing_status': True}
        }
        
        # Act & Assert - Should not raise exceptions
        try:
            self.validator.print_validation_report(malformed_results)
        except Exception as e:
            self.fail(f"print_validation_report should handle malformed results gracefully: {e}")


class TestEnvironmentValidatorEdgeCases(unittest.TestCase):
    """Test edge cases in environment validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = EnvironmentValidator()
    
    def test_validate_environment_with_empty_string_values(self):
        """Test environment validation with empty string values."""
        # Arrange & Act
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': '',
            'CLAUDE_PAT': ''
        }):
            results = self.validator.validate_environment()
        
        # Assert
        # Empty strings should be treated as missing or invalid
        self.assertEqual(results['ANTHROPIC_API_KEY']['status'], ValidationStatus.MISSING)
        self.assertEqual(results['CLAUDE_PAT']['status'], ValidationStatus.MISSING)
    
    def test_validate_environment_with_whitespace_values(self):
        """Test environment validation with whitespace-only values."""
        # Arrange & Act
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': '   ',
            'CLAUDE_PAT': '\t\n'
        }):
            results = self.validator.validate_environment()
        
        # Assert
        # Whitespace-only values should be treated as missing or invalid
        self.assertEqual(results['ANTHROPIC_API_KEY']['status'], ValidationStatus.MISSING)
        self.assertEqual(results['CLAUDE_PAT']['status'], ValidationStatus.MISSING)
    
    def test_custom_secret_config_without_validation_function(self):
        """Test secret configuration without validation function."""
        # Arrange
        custom_config = SecretConfig(
            name="CUSTOM_SECRET",
            description="Custom secret without validation",
            validation_func=None
        )
        
        validator = EnvironmentValidator()
        validator.secrets_config.append(custom_config)
        
        # Act
        with patch.dict(os.environ, {'CUSTOM_SECRET': 'any_value'}):
            results = validator.validate_environment()
        
        # Assert
        # Should handle secrets without validation functions
        self.assertIn('CUSTOM_SECRET', results)
        # Without validation function, should be marked as available if present
        self.assertEqual(results['CUSTOM_SECRET']['status'], ValidationStatus.AVAILABLE)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)